import math
import torch
import torch.nn as nn
from snn.spike_function import SpikeFunction


class ParametricLIF(nn.Module):
    """
    Learnable Leaky Integrate-and-Fire (PLIF) neuron layer.
    Features:
    - Channel/Feature-wise learnable membrane decay (beta) bounded in (0, 1)
    - Soft Reset (reset-by-subtraction) to preserve above-threshold excess potential
    - Detached reset spike to prevent gradient oscillations during BPTT
    """
    def __init__(self, channels_or_features: int, init_beta: float = 0.85, threshold: float = 1.0):
        super().__init__()
        self.threshold = threshold
        # Inverse sigmoid initialization so sigmoid(w) == init_beta
        init_w = math.log(init_beta / (1.0 - init_beta))
        self.w = nn.Parameter(torch.full((channels_or_features,), init_w, dtype=torch.float32))

    @property
    def beta(self) -> torch.Tensor:
        return torch.sigmoid(self.w)

    def forward(self, cur: torch.Tensor, membrane: torch.Tensor):
        if cur.dim() == 4:
            beta = self.beta.view(1, -1, 1, 1)
        else:
            beta = self.beta.view(1, -1)

        # 1. Leaky integration
        membrane = beta * membrane + cur

        # 2. Surrogate spike generation
        v_th = torch.tensor(self.threshold, device=membrane.device, dtype=membrane.dtype)
        spike = SpikeFunction.apply(membrane, v_th)

        # 3. Soft reset with detached spike
        membrane = membrane - spike.detach() * self.threshold

        return spike, membrane


class SpikingDuelingDQN(nn.Module):
    """
    Spiking Dueling Deep Q-Network for ViZDoom (84x84 spatial resolution).
    """
    def __init__(
        self,
        input_channels: int = 1,
        num_actions: int = 3,
        time_steps: int = 4,
        threshold: float = 1.0,
        readout_decay: float = 0.85,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.num_actions = num_actions
        self.time_steps = time_steps
        self.readout_decay = readout_decay

        # Spatial Stem: 84x84 -> 20x20
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.lif1 = ParametricLIF(32, init_beta=0.80, threshold=threshold)

        # Mid-level Features: 20x20 -> 9x9
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.lif2 = ParametricLIF(64, init_beta=0.85, threshold=threshold)

        # High-level Spatial Features: 9x9 -> 7x7
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(64)
        self.lif3 = ParametricLIF(64, init_beta=0.90, threshold=threshold)

        # Spiking Association Layer: 64 * 7 * 7 = 3136 -> 512
        self.fc1 = nn.Linear(3136, 512, bias=False)
        self.bn_fc = nn.BatchNorm1d(512)
        self.lif_fc = ParametricLIF(512, init_beta=0.90, threshold=threshold)

        # Dueling Non-Spiking Membrane Readout Heads
        self.value_head = nn.Linear(512, 1)
        self.advantage_head = nn.Linear(512, num_actions)

        self._init_weights()

    def _init_weights(self):
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
        Args:
            x: Input tensor of shape:
               - (B, C, 84, 84) for static direct encoding across T steps
               - (B, T, C, 84, 84) for temporal sequence encoding
            return_firing_rate: If True, returns (q_values, firing_rate_tensor)
                                where firing_rate_tensor is differentiable for sparsity loss.
        """
        # Normalize input to [0, 1] if passed as uint8 or unscaled [0, 255]
        if x.dtype == torch.uint8:
            x = x.float() / 255.0
        elif x.max() > 1.5:
            x = x / 255.0

        is_temporal_sequence = (x.dim() == 5)
        B = x.size(0)
        T = x.size(1) if is_temporal_sequence else self.time_steps
        device = x.device
        dtype = x.dtype

        # Initialize membrane potentials
        mem1 = torch.zeros(B, 32, 20, 20, device=device, dtype=dtype)
        mem2 = torch.zeros(B, 64, 9, 9, device=device, dtype=dtype)
        mem3 = torch.zeros(B, 64, 7, 7, device=device, dtype=dtype)
        mem_fc = torch.zeros(B, 512, device=device, dtype=dtype)

        mem_val = torch.zeros(B, 1, device=device, dtype=dtype)
        mem_adv = torch.zeros(B, self.num_actions, device=device, dtype=dtype)

        # Fast-path: pre-compute Conv1 + BN1 once if input is a static frame
        if not is_temporal_sequence:
            cur1_static = self.bn1(self.conv1(x))

        spike_sum = torch.zeros(1, device=device, dtype=dtype)
        neurons_per_step = float(B * (32 * 20 * 20 + 64 * 9 * 9 + 64 * 7 * 7 + 512))

        for t in range(T):
            cur1 = self.bn1(self.conv1(x[:, t])) if is_temporal_sequence else cur1_static
            spk1, mem1 = self.lif1(cur1, mem1)

            cur2 = self.bn2(self.conv2(spk1))
            spk2, mem2 = self.lif2(cur2, mem2)

            cur3 = self.bn3(self.conv3(spk2))
            spk3, mem3 = self.lif3(cur3, mem3)

            flat = spk3.flatten(1)
            cur_fc = self.bn_fc(self.fc1(flat))
            spk_fc, mem_fc = self.lif_fc(cur_fc, mem_fc)

            # Leaky membrane integration in the non-spiking readout layer
            mem_val = self.readout_decay * mem_val + self.value_head(spk_fc)
            mem_adv = self.readout_decay * mem_adv + self.advantage_head(spk_fc)

            if return_firing_rate:
                # Keep attached to computation graph so homeostatic loss can backpropagate
                spike_sum = spike_sum + spk1.sum() + spk2.sum() + spk3.sum() + spk_fc.sum()

        # Dueling Q-value aggregation: Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        q_values = mem_val + (mem_adv - mem_adv.mean(dim=1, keepdim=True))

        if return_firing_rate:
            firing_rate = spike_sum / (neurons_per_step * T)
            return q_values, firing_rate.squeeze(0)

        return q_values