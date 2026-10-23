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

    Silicon Implementation:
        Hardware does not contain floating-point multipliers for decay.
        Instead, membrane leak is computed via bitwise arithmetic right-shifts:
            V[t] = V[t-1] - (V[t-1] >> k) + I[t]
        which is mathematically equivalent to:
            V[t] = (1 - 2^(-k)) * V[t-1] + I[t]

    Shift Options:
        k = 1 -> beta = 1 - 0.5000 = 0.5000
        k = 2 -> beta = 1 - 0.2500 = 0.7500
        k = 3 -> beta = 1 - 0.1250 = 0.8750  (Default)
        k = 4 -> beta = 1 - 0.0625 = 0.9375
    """
    def __init__(self, shift_k: int = 3, threshold: float = 1.0):
        super().__init__()
        assert shift_k in (1, 2, 3, 4), f"SNAP-V supports shift_k in {{1, 2, 3, 4}}, got {shift_k}"
        self.shift_k = shift_k
        self.threshold = threshold
        # Exact power-of-two decay factor
        self.beta = 1.0 - (2.0 ** (-shift_k))

    def forward(self, cur: torch.Tensor, membrane: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Executes one micro-timestep on the neuromorphic core.
        
        Args:
            cur: Incoming synaptic input current I[t].
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
        # SNAP-V hardware subtracts V_th on spike, preserving residual above-threshold voltage
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
    # Forward pass: quantized into discrete integer levels and scaled back
    w_quant = (weight / scale).round().clamp(-128, 127) * scale
    # STE: Gradients bypass rounding and clamp directly into full-precision weight
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
# 3. Hardware-Aware SNAP-V Spiking Dueling DQN
# =============================================================================

class SnapVSpikingDQN(nn.Module):
    """
    Hardware-Constrained Spiking Dueling DQN for SNAP-V (Cerebra-H) FPGA.

    Constraints & Specifications:
        - Input: (B, 4, 32, 32) (4-frame stacked grayscale at 32x32 resolution)
        - Actions: 5 Compound Actions
        - Simulation Timesteps: T = 4 micro-ticks
        - Conv1: 4 -> 16 channels, 4x4 kernel, stride=2 -> 16x15x15 = 3,600 neurons | ShiftLIF(k=2, beta=0.75)
        - Conv2: 16 -> 24 channels, 3x3 kernel, stride=2 -> 24x7x7 = 1,176 neurons  | ShiftLIF(k=3, beta=0.875)
        - FC1: 1,176 -> 128 decision neurons                                         | ShiftLIF(k=3, beta=0.875)
        - Dueling Heads: value_head (128 -> 1), advantage_head (128 -> 5)
        - Total Spiking Neurons: 3,600 + 1,176 + 128 = 4,904 neurons
        - Total Synaptic Weights: ~155,776 synapses (fits within SNAP-V 524,288 SRAM capacity)
    """
    def __init__(
        self,
        input_channels: int = 4,
        num_actions: int = 5,
        time_steps: int = 4,
        threshold: float = 1.0,
        readout_shift_k: int = 3,  # beta = 1 - 2^(-3) = 0.875
    ):
        super().__init__()
        self.input_channels = input_channels
        self.num_actions = num_actions
        self.time_steps = time_steps
        self.threshold = threshold
        self.readout_beta = 1.0 - (2.0 ** (-readout_shift_k))

        # Layer 1: Conv1 (4 -> 16, 4x4, stride 2) -> (B, 16, 15, 15) = 3,600 neurons
        # (32 - 4) / 2 + 1 = 15
        self.conv1 = QuantConv2d(input_channels, 16, kernel_size=4, stride=2, bias=False)
        self.lif1 = ShiftLIF(shift_k=2, threshold=threshold)  # beta = 0.75

        # Layer 2: Conv2 (16 -> 24, 3x3, stride 2) -> (B, 24, 7, 7) = 1,176 neurons
        # (15 - 3) / 2 + 1 = 7
        self.conv2 = QuantConv2d(16, 24, kernel_size=3, stride=2, bias=False)
        self.lif2 = ShiftLIF(shift_k=3, threshold=threshold)  # beta = 0.875

        # Layer 3: FC1 (1,176 -> 128) -> 128 decision neurons
        # Flattened spatial shape: 24 * 7 * 7 = 1,176
        self.fc1 = QuantLinear(1176, 128, bias=False)
        self.lif_fc = ShiftLIF(shift_k=3, threshold=threshold)  # beta = 0.875

        # Dueling Readout Heads (Non-spiking leaky accumulator)
        self.value_head = QuantLinear(128, 1, bias=True)
        self.advantage_head = QuantLinear(128, num_actions, bias=True)

        self._init_weights()

    def _init_weights(self):
        """Initializes weights for stable neuromorphic spike propagation."""
        for m in self.modules():
            if isinstance(m, (QuantConv2d, QuantLinear, nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def count_resources(self) -> dict:
        """Computes exact hardware resource usage for SNAP-V deployment."""
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

    def forward(self, x: torch.Tensor, return_firing_rate: bool = False):
        """
        Runs T=4 hardware simulation timesteps on input batch x.
        Args:
            x: Tensor of shape (B, 4, 32, 32)
            return_firing_rate: If True, returns (q_values, firing_rate)
        """
        # Ensure input is float normalized in [0, 1]
        if x.dtype == torch.uint8:
            x = x.float() / 255.0
        elif x.max() > 1.5:
            x = x / 255.0

        B = x.size(0)
        T = self.time_steps
        device = x.device
        dtype = x.dtype

        # -------------------------------------------------------------
        # Initialize Membrane Potentials V[0] = 0
        # -------------------------------------------------------------
        mem1 = torch.zeros(B, 16, 15, 15, device=device, dtype=dtype)
        mem2 = torch.zeros(B, 24, 7, 7, device=device, dtype=dtype)
        mem_fc = torch.zeros(B, 128, device=device, dtype=dtype)

        # Readout integrators
        mem_val = torch.zeros(B, 1, device=device, dtype=dtype)
        mem_adv = torch.zeros(B, self.num_actions, device=device, dtype=dtype)

        # Static temporal input: Pre-compute Conv1 current once for all T ticks
        cur1_static = self.conv1(x)

        spike_sum = torch.zeros(1, device=device, dtype=dtype)
        total_neurons = float(B * (16 * 15 * 15 + 24 * 7 * 7 + 128))

        # =============================================================
        # Step through T=4 micro-timesteps
        # =============================================================
        for t in range(T):
            # Conv1
            spk1, mem1 = self.lif1(cur1_static, mem1)

            # Conv2
            cur2 = self.conv2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)

            # FC1
            flat = spk2.flatten(1)
            cur_fc = self.fc1(flat)
            spk_fc, mem_fc = self.lif_fc(cur_fc, mem_fc)

            # Shift-based Readout accumulation: V_out[t] = beta * V_out[t-1] + cur
            mem_val = self.readout_beta * mem_val + self.value_head(spk_fc)
            mem_adv = self.readout_beta * mem_adv + self.advantage_head(spk_fc)

            if return_firing_rate:
                spike_sum = spike_sum + spk1.sum() + spk2.sum() + spk_fc.sum()

        # Dueling Q-value combination: Q(s, a) = V(s) + (A(s, a) - mean(A(s, a)))
        q_values = mem_val + (mem_adv - mem_adv.mean(dim=1, keepdim=True))

        if return_firing_rate:
            firing_rate = spike_sum / (total_neurons * T)
            return q_values, firing_rate.squeeze(0)

        return q_values
