"""
SNAP-V (Cerebra-H) Hardware-Constrained Unified SNN Architecture
================================================================
Combines:
  1. 4-frame visual observation (4, 32, 32) via INT8 QuantConv2d feature pyramid.
  2. 4-dimensional normalized scalar telemetry (health, ammo, ready, motion energy)
     injected directly into associative layer FC1.
  3. 8-action tactical compound action space (turning, advancing, strafing, strafe-firing).
  4. Bit-shift Leaky Integrate-and-Fire (ShiftLIF) dynamics with power-of-two decay.
  5. Dueling Q-network readout with leaky accumulation.

Designed strictly for compatibility with the SNAP-V neuromorphic FPGA core.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from snn.spike_function import SpikeFunction


# =============================================================================
# 1. Hardware Shift-Based Leaky Integrate-and-Fire Neuron (ShiftLIF)
# =============================================================================

class ShiftLIF(nn.Module):
    """
    Hardware-accurate Leaky Integrate-and-Fire (LIF) neuron layer for SNAP-V / Cerebra-H.
    Membrane leak is computed via bitwise arithmetic right-shifts:
        V[t] = V[t-1] - (V[t-1] >> k) + I[t] == (1 - 2^(-k)) * V[t-1] + I[t]

    Shift Options:
        k = 1 -> beta = 0.5000
        k = 2 -> beta = 0.7500 (Used in Conv1 for fast visual transients)
        k = 3 -> beta = 0.8750 (Used in Conv2, FC1, and Readout for working memory)
        k = 4 -> beta = 0.9375
    """
    def __init__(self, shift_k: int = 3, threshold: float = 1.0):
        super().__init__()
        assert shift_k in (1, 2, 3, 4), f"SNAP-V supports shift_k in {{1, 2, 3, 4}}, got {shift_k}"
        self.shift_k = shift_k
        self.threshold = threshold
        self.beta = 1.0 - (2.0 ** (-shift_k))

    def forward(self, cur: torch.Tensor, membrane: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Executes one micro-timestep on the neuromorphic core.
        
        Args:
            cur: Synaptic input current I[t].
            membrane: Prior membrane potential V[t-1].
            
        Returns:
            spike: Binary spike output tensor in {0.0, 1.0}.
            membrane: Updated membrane potential V[t] after leak and soft reset.
        """
        # 1. Leaky Integration: V[t] = beta * V[t-1] + I[t]
        membrane = self.beta * membrane + cur

        # 2. Spike Generation using Surrogate Gradient
        v_th = torch.tensor(self.threshold, device=membrane.device, dtype=membrane.dtype)
        spike = SpikeFunction.apply(membrane, v_th)

        # 3. Hardware Soft Reset (Reset-by-Subtraction):
        # Subtracts V_th on spike, preserving residual charge above threshold
        membrane = membrane - spike.detach() * self.threshold

        return spike, membrane

    def extra_repr(self) -> str:
        return f"shift_k={self.shift_k}, beta={self.beta:.4f}, threshold={self.threshold}"


# =============================================================================
# 2. Quantization-Aware Fake Quantization (STE) Layers (INT8)
# =============================================================================

def fake_quantize_int8(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Simulates SNAP-V 8-bit signed integer synaptic weights [-128, 127]
    using Straight-Through Estimator (STE) for backward gradient flow.
    """
    max_val = weight.abs().max()
    scale = torch.clamp(max_val / 127.0, min=1e-8)
    w_quant = (weight / scale).round().clamp(-128, 127) * scale
    w_ste = weight + (w_quant - weight).detach()
    return w_ste, scale


class QuantConv2d(nn.Conv2d):
    """Conv2D layer with fake INT8 quantization for SNAP-V SRAM storage."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size, stride=1, padding=0, bias: bool = False):
        super().__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        self.register_buffer("weight_scale", torch.tensor(1.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_ste, scale = fake_quantize_int8(self.weight)
        self.weight_scale.copy_(scale.detach())
        return F.conv2d(x, w_ste, self.bias, self.stride, self.padding, self.dilation, self.groups)


class QuantLinear(nn.Linear):
    """Linear layer with fake INT8 quantization for SNAP-V SRAM storage."""
    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__(in_features, out_features, bias=bias)
        self.register_buffer("weight_scale", torch.tensor(1.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_ste, scale = fake_quantize_int8(self.weight)
        self.weight_scale.copy_(scale.detach())
        return F.linear(x, w_ste, self.bias)


# =============================================================================
# 3. Hardware-Aware Unified SNAP-V Spiking Dueling DQN
# =============================================================================

class SnapVUnifiedDQN(nn.Module):
    """
    Hardware-Constrained Multi-Modal Spiking Dueling DQN for SNAP-V (Cerebra-H).

    Inputs:
        x_vis: Visual 4-frame grayscale stack of shape (B, 4, 32, 32)
        x_tele: 4-dimensional normalized scalar telemetry of shape (B, 4):
            [health_norm, ammo_norm, attack_ready, motion_energy]

    Outputs:
        q_values: State-action value estimates of shape (B, 8)
            0: TURN_LEFT
            1: TURN_RIGHT
            2: MOVE_FORWARD
            3: STRAFE_LEFT
            4: STRAFE_RIGHT
            5: ATTACK
            6: STRAFE_LEFT_ATTACK
            7: STRAFE_RIGHT_ATTACK

    Cluster Mapping & Synapse Budget:
        - Conv1: 4 -> 16, 4x4, s=2 -> 3,600 neurons | 1,024 synapses  (Clusters 00-19)
        - Conv2: 16 -> 24, 3x3, s=2 -> 1,176 neurons | 3,456 synapses  (Clusters 20-27)
        - FC1: (1,176 visual + 4 telemetry = 1,180) -> 128 neurons | 151,040 synapses (Clusters 28-30)
        - Readout: Value (128 -> 1) + Advantage (128 -> 8) | 1,152 synapses (Cluster 31)
        Total Synapses: 156,672 INT8 synapses (29.88% of SNAP-V 524,288 SRAM capacity)
    """
    def __init__(
        self,
        input_channels: int = 4,
        telemetry_dim: int = 4,
        num_actions: int = 8,
        time_steps: int = 4,
        threshold: float = 1.0,
        readout_shift_k: int = 3,  # beta = 1 - 2^(-3) = 0.875
    ):
        super().__init__()
        self.input_channels = input_channels
        self.telemetry_dim = telemetry_dim
        self.num_actions = num_actions
        self.time_steps = time_steps
        self.threshold = threshold
        self.readout_beta = 1.0 - (2.0 ** (-readout_shift_k))

        # Visual Stem: Layer 1 (Conv1) -> (B, 16, 15, 15) = 3,600 neurons
        self.conv1 = QuantConv2d(input_channels, 16, kernel_size=4, stride=2, bias=False)
        self.lif1 = ShiftLIF(shift_k=2, threshold=threshold)  # beta = 0.750

        # Visual Stem: Layer 2 (Conv2) -> (B, 24, 7, 7) = 1,176 neurons
        self.conv2 = QuantConv2d(16, 24, kernel_size=3, stride=2, bias=False)
        self.lif2 = ShiftLIF(shift_k=3, threshold=threshold)  # beta = 0.875

        # Associative Layer: FC1 (1,176 visual spikes + telemetry_dim scalars -> 128 neurons)
        self.visual_features_dim = 24 * 7 * 7  # 1,176
        self.fc1_in_features = self.visual_features_dim + telemetry_dim  # 1,180
        self.fc1 = QuantLinear(self.fc1_in_features, 128, bias=False)
        self.lif_fc = ShiftLIF(shift_k=3, threshold=threshold)  # beta = 0.875

        # Dueling Readout Heads (Non-spiking leaky accumulator in Cluster 31)
        self.value_head = QuantLinear(128, 1, bias=True)
        self.advantage_head = QuantLinear(128, num_actions, bias=True)

        self._init_weights()

    def _init_weights(self):
        """Initializes weights with Kaiming normal distribution for stable spiking."""
        for m in self.modules():
            if isinstance(m, (QuantConv2d, QuantLinear, nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def count_resources(self) -> dict:
        """Computes exact hardware resource footprint for SNAP-V deployment."""
        neurons = {
            "conv1_neurons": 16 * 15 * 15,
            "conv2_neurons": 24 * 7 * 7,
            "fc1_neurons": 128,
            "readout_neurons": 1 + self.num_actions,
            "total_spiking_neurons": 16 * 15 * 15 + 24 * 7 * 7 + 128,
        }

        synapses = {
            "conv1_synapses": self.conv1.weight.numel(),
            "conv2_synapses": self.conv2.weight.numel(),
            "fc1_synapses": self.fc1.weight.numel(),
            "fc1_visual_synapses": 128 * self.visual_features_dim,
            "fc1_telemetry_synapses": 128 * self.telemetry_dim,
            "value_synapses": self.value_head.weight.numel(),
            "advantage_synapses": self.advantage_head.weight.numel(),
            "total_synapses": (
                self.conv1.weight.numel()
                + self.conv2.weight.numel()
                + self.fc1.weight.numel()
                + self.value_head.weight.numel()
                + self.advantage_head.weight.numel()
            ),
        }
        return {"neurons": neurons, "synapses": synapses}

    def forward(
        self,
        x_vis: torch.Tensor,
        x_tele: torch.Tensor,
        return_firing_rate: bool = False,
    ):
        """
        Executes T=4 hardware simulation micro-timesteps.

        Args:
            x_vis: Visual tensor of shape (B, 4, 32, 32).
            x_tele: Telemetry tensor of shape (B, 4) in [0.0, 1.0].
            return_firing_rate: If True, returns (q_values, firing_rate).

        Returns:
            q_values: Estimated state-action values of shape (B, 8).
        """
        # Ensure visual input is float normalized in [0, 1]
        if x_vis.dtype == torch.uint8:
            x_vis = x_vis.float() / 255.0
        elif x_vis.max() > 1.5:
            x_vis = x_vis / 255.0

        if x_tele.dtype != x_vis.dtype:
            x_tele = x_tele.to(dtype=x_vis.dtype)

        B = x_vis.size(0)
        T = self.time_steps
        device = x_vis.device
        dtype = x_vis.dtype

        # -----------------------------------------------------------------
        # Explicit Membrane State Reset (Zero state leak between forward calls)
        # -----------------------------------------------------------------
        mem1 = torch.zeros(B, 16, 15, 15, device=device, dtype=dtype)
        mem2 = torch.zeros(B, 24, 7, 7, device=device, dtype=dtype)
        mem_fc = torch.zeros(B, 128, device=device, dtype=dtype)

        # Readout leaky integrators
        mem_val = torch.zeros(B, 1, device=device, dtype=dtype)
        mem_adv = torch.zeros(B, self.num_actions, device=device, dtype=dtype)

        # Static Conv1 visual current (precomputed once for all T ticks)
        cur1_static = self.conv1(x_vis)

        spike_sum = torch.zeros(1, device=device, dtype=dtype)
        total_neurons = float(B * (16 * 15 * 15 + 24 * 7 * 7 + 128))

        # =================================================================
        # Unroll T=4 micro-timesteps on SNN core
        # =================================================================
        for t in range(T):
            # 1. Conv1: Integrate static input current -> Spikes (B, 16, 15, 15)
            spk1, mem1 = self.lif1(cur1_static, mem1)

            # 2. Conv2: Integrate Conv1 spikes -> Spikes (B, 24, 7, 7)
            cur2 = self.conv2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)

            # 3. Associative Fusion: Concatenate visual spikes (1,176) with telemetry (4)
            flat_spk = spk2.flatten(1)  # (B, 1176)
            fused_input = torch.cat([flat_spk, x_tele], dim=1)  # (B, 1180)

            # 4. FC1: Compute associative synaptic current -> Spikes (B, 128)
            cur_fc = self.fc1(fused_input)
            spk_fc, mem_fc = self.lif_fc(cur_fc, mem_fc)

            # 5. Readout Leaky Accumulator (Cluster 31):
            # V_out[t] = beta * V_out[t-1] + W * S_fc[t]
            mem_val = self.readout_beta * mem_val + self.value_head(spk_fc)
            mem_adv = self.readout_beta * mem_adv + self.advantage_head(spk_fc)

            if return_firing_rate:
                spike_sum = spike_sum + spk1.sum() + spk2.sum() + spk_fc.sum()

        # Dueling Q-value combination: Q(s, a) = V(s) + (A(s, a) - mean(A(s, :)))
        q_values = mem_val + (mem_adv - mem_adv.mean(dim=1, keepdim=True))

        if return_firing_rate:
            firing_rate = spike_sum / (total_neurons * T)
            return q_values, firing_rate.squeeze(0)

        return q_values
