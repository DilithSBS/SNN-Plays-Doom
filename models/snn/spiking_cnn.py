import math
import torch
import torch.nn as nn
from snn.spike_function import SpikeFunction


class ParametricLIF(nn.Module):
    """
    Parametric Leaky Integrate-and-Fire (PLIF) neuron layer.

    Biological Analogy:
        Each neuron acts like a leaky capacitor. Input current charges the membrane
        potential. If it reaches the firing threshold, an action potential (spike) is
        emitted, and the voltage drops back down (resets).

    Key Features:
        - Learnable Beta (decay): Each channel learns its own decay rate via gradient descent.
        - Soft Reset (by subtraction): Preserves excess voltage above threshold (V - V_th)
          rather than resetting to zero, retaining sub-millisecond temporal precision.
        - Detached Reset: Stops gradient looping through the reset operation during BPTT.
    """
    def __init__(self, channels_or_features: int, init_beta: float = 0.85, threshold: float = 1.0):
        super().__init__()
        # Firing threshold (V_th): Voltage needed to trigger a spike (typically 1.0)
        self.threshold = threshold

        # Learnable decay parameterization:
        # Beta must stay strictly between 0 and 1. We store an unconstrained weight 'w'
        # and compute beta = sigmoid(w).
        # We use inverse sigmoid (logit) so that sigmoid(init_w) exactly equals init_beta:
        # logit(p) = ln(p / (1 - p))
        init_w = math.log(init_beta / (1.0 - init_beta))
        self.w = nn.Parameter(torch.full((channels_or_features,), init_w, dtype=torch.float32))

    @property
    def beta(self) -> torch.Tensor:
        """Computes the effective membrane retention factor in (0, 1)."""
        return torch.sigmoid(self.w)

    def forward(self, cur: torch.Tensor, membrane: torch.Tensor):
        """
        Executes one micro-timestep for the LIF neuron layer.

        Args:
            cur: Incoming synaptic input current from previous layer.
                 - For Conv layers: Shape (B, C, H, W)
                 - For Linear layers: Shape (B, Features)
            membrane: Previous membrane potential V[t-1] with same shape as cur.

        Returns:
            spike: Binary tensor (0.0 or 1.0) indicating emitted spikes at timestep t.
            membrane: Updated membrane potential V[t] after leaky integration and reset.
        """
        # Reshape beta for broadcasting across spatial dimensions:
        # Conv4D: (1, Channels, 1, 1) | Linear2D: (1, Features)
        if cur.dim() == 4:
            beta = self.beta.view(1, -1, 1, 1)
        else:
            beta = self.beta.view(1, -1)

        # 1. Leaky Integration: V[t] = beta * V[t-1] + Input_Current[t]
        membrane = beta * membrane + cur

        # 2. Spike Generation with Surrogate Gradient:
        # Spike = 1.0 if membrane >= threshold, else 0.0
        v_th = torch.tensor(self.threshold, device=membrane.device, dtype=membrane.dtype)
        spike = SpikeFunction.apply(membrane, v_th)

        # 3. Soft Reset (Reset-by-Subtraction):
        # Subtract threshold from neurons that spiked, keeping any leftover excess charge.
        # .detach() prevents backpropagating through the discontinuous reset jump.
        membrane = membrane - spike.detach() * self.threshold

        return spike, membrane


class SpikingDuelingDQN(nn.Module):
    """
    Spiking Dueling Deep Q-Network for ViZDoom (84x84 grayscale frames).

    Architecture:
        1. Visual Stem: 3 Spiking Convolutional layers (Conv2D + BatchNorm + PLIF).
        2. Association Area: 1 Spiking Fully Connected layer (Linear + BatchNorm + PLIF).
        3. Dueling Readout: Two parallel non-spiking membrane accumulators:
           - Value Stream V(s): Scalar estimate of how good the current game state is.
           - Advantage Stream A(s, a): Vector estimate of each action's relative merit.
        4. Aggregation: Q(s, a) = V(s) + (A(s, a) - mean(A(s, a)))
    """
    def __init__(
        self,
        input_channels: int = 1,     # Grayscale frame = 1 channel
        num_actions: int = 3,        # ViZDoom actions: [MOVE_LEFT, MOVE_RIGHT, ATTACK]
        time_steps: int = 4,         # Micro-timesteps (T) to simulate per game frame
        threshold: float = 1.0,      # Neuron firing threshold
        readout_decay: float = 0.85, # Decay factor for non-spiking output integrator
    ):
        super().__init__()
        self.input_channels = input_channels
        self.num_actions = num_actions
        self.time_steps = time_steps
        self.readout_decay = readout_decay

        # --- Layer 1 (Conv1): Low-level visual features (edges, contrast) ---
        # Input: (B, 1, 84, 84) -> Output: (B, 32, 20, 20)
        # Spatial formula: (84 - 8) / 4 + 1 = 20
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.lif1 = ParametricLIF(32, init_beta=0.80, threshold=threshold)

        # --- Layer 2 (Conv2): Mid-level features (corners, textures) ---
        # Input: (B, 32, 20, 20) -> Output: (B, 64, 9, 9)
        # Spatial formula: (20 - 4) / 2 + 1 = 9
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.lif2 = ParametricLIF(64, init_beta=0.85, threshold=threshold)

        # --- Layer 3 (Conv3): High-level spatial shapes (monster silhouette, walls) ---
        # Input: (B, 64, 9, 9) -> Output: (B, 64, 7, 7)
        # Spatial formula: (9 - 3) / 1 + 1 = 7
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(64)
        self.lif3 = ParametricLIF(64, init_beta=0.90, threshold=threshold)

        # --- Layer 4 (FC1): Spiking Association Area ---
        # Flattened size: 64 channels * 7 height * 7 width = 3136 neurons
        # Output: 512 decision neurons
        self.fc1 = nn.Linear(3136, 512, bias=False)
        self.bn_fc = nn.BatchNorm1d(512)
        self.lif_fc = ParametricLIF(512, init_beta=0.90, threshold=threshold)

        # --- Dueling Readout Heads (Non-Spiking Membrane Integrators) ---
        # Value Stream: Evaluates overall state quality V(s) -> (B, 1)
        self.value_head = nn.Linear(512, 1)
        # Advantage Stream: Evaluates relative advantage of each action A(s, a) -> (B, num_actions)
        self.advantage_head = nn.Linear(512, num_actions)

        # Initialize network weights for stable spiking dynamics
        self._init_weights()

    def _init_weights(self):
        """Kaiming normal initialization for conv/linear layers and standard BN initialization."""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, return_firing_rate: bool = False):
        """
        Simulates the SNN across T timesteps and returns action Q-values.

        Args:
            x: Input frame tensor. Supports two shapes:
               - (B, C, 84, 84): Single game frame (direct temporal encoding: repeated for T steps).
               - (B, T, C, 84, 84): Temporal sequence across T steps.
            return_firing_rate: If True, also returns the differentiable network firing rate
                                used by the training loop for homeostatic sparsity regularization.

        Returns:
            q_values: Tensor of shape (B, num_actions) containing estimated Q(s, a).
            firing_rate (optional): Scalar tensor measuring the fraction of neurons that fired.
        """
        # Ensure pixel values are normalized floats in [0.0, 1.0]
        if x.dtype == torch.uint8:
            x = x.float() / 255.0
        elif x.max() > 1.5:
            x = x / 255.0

        is_temporal_sequence = (x.dim() == 5)
        B = x.size(0)
        T = x.size(1) if is_temporal_sequence else self.time_steps
        device = x.device
        dtype = x.dtype

        # -------------------------------------------------------------
        # Initialize Membrane Potentials V[0] = 0 for all spiking layers
        # -------------------------------------------------------------
        mem1 = torch.zeros(B, 32, 20, 20, device=device, dtype=dtype)
        mem2 = torch.zeros(B, 64, 9, 9, device=device, dtype=dtype)
        mem3 = torch.zeros(B, 64, 7, 7, device=device, dtype=dtype)
        mem_fc = torch.zeros(B, 512, device=device, dtype=dtype)

        # Non-spiking readout membrane accumulators
        mem_val = torch.zeros(B, 1, device=device, dtype=dtype)
        mem_adv = torch.zeros(B, self.num_actions, device=device, dtype=dtype)

        # Optimization: If the input is static across time, pre-compute Conv1+BN1 once
        # instead of repeating identical convolutions T times
        if not is_temporal_sequence:
            cur1_static = self.bn1(self.conv1(x))

        # Accumulator for tracking network-wide spike activity (differentiable)
        spike_sum = torch.zeros(1, device=device, dtype=dtype)
        neurons_per_step = float(B * (32 * 20 * 20 + 64 * 9 * 9 + 64 * 7 * 7 + 512))

        # =============================================================
        # Temporal Simulation Loop: Step through micro-timesteps t = 0..T-1
        # =============================================================
        for t in range(T):
            # Conv1: Process raw image pixels into first spike train
            cur1 = self.bn1(self.conv1(x[:, t])) if is_temporal_sequence else cur1_static
            spk1, mem1 = self.lif1(cur1, mem1)

            # Conv2: Mid-level feature detection driven by spk1 pulses
            cur2 = self.bn2(self.conv2(spk1))
            spk2, mem2 = self.lif2(cur2, mem2)

            # Conv3: High-level shape detection driven by spk2 pulses
            cur3 = self.bn3(self.conv3(spk2))
            spk3, mem3 = self.lif3(cur3, mem3)

            # FC1: Flatten spatial grid (3136) and feed decision neurons
            flat = spk3.flatten(1)
            cur_fc = self.bn_fc(self.fc1(flat))
            spk_fc, mem_fc = self.lif_fc(cur_fc, mem_fc)

            # Non-Spiking Readout: Leaky accumulation of Value and Advantage
            # V_out[t] = readout_decay * V_out[t-1] + W * spikes[t]
            mem_val = self.readout_decay * mem_val + self.value_head(spk_fc)
            mem_adv = self.readout_decay * mem_adv + self.advantage_head(spk_fc)

            if return_firing_rate:
                # Sum all emitted spikes across all 4 layers at timestep t
                spike_sum = spike_sum + spk1.sum() + spk2.sum() + spk3.sum() + spk_fc.sum()

        # =============================================================
        # Dueling Q-Value Aggregation:
        # Q(s, a) = V(s) + (A(s, a) - (1 / |A|) * sum(A(s, a')))
        # Subtracting the mean advantage stabilizes learning and resolves identifiability.
        # =============================================================
        q_values = mem_val + (mem_adv - mem_adv.mean(dim=1, keepdim=True))

        if return_firing_rate:
            # Overall firing rate = total spikes / (total neurons * T timesteps)
            firing_rate = spike_sum / (neurons_per_step * T)
            return q_values, firing_rate.squeeze(0)

        return q_values