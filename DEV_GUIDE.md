# Developer & Architecture Guide: SNN-Plays-Doom

This guide provides an in-depth technical explanation of the neuromorphic design, mathematical formulations, training mechanics, hardware-constrained compilation, and codebase architecture in **SNN-Plays-Doom**.

---

## 1. Neuromorphic Fundamentals & SNN Mathematics

Traditional Artificial Neural Networks (ANNs) process information as continuous floating-point activations evaluated in a single forward pass. In contrast, **Spiking Neural Networks (SNNs)** compute temporally using discrete event-based binary spikes ($S \in \{0, 1\}$).

### 1.1 Leaky Integrate-and-Fire (LIF) Dynamics

Every spiking neuron models the electrical properties of a biological cell membrane with a resistor-capacitor (RC) circuit. Over discretized time steps $t \in \{1, \dots, T\}$, incoming synaptic current $I[t]$ charges the membrane potential $V[t]$. Between input pulses, charge leaks through the membrane resistance:

$$V[t] = \beta \cdot V[t-1] + I[t]$$

Where:
- $V[t]$ is the membrane potential at simulation time step $t$.
- $I[t]$ is the input current produced by synaptic weights and batch normalization.
- $\beta \in (0, 1)$ is the membrane decay constant ($\beta = e^{-\Delta t / \tau_m}$).

### 1.2 Learnable Membrane Decay (Parametric LIF)

In the unconstrained software baseline, `ParametricLIF` learns an optimal decay rate per channel/feature:

$$\beta = \sigma(w) = \frac{1}{1 + e^{-w}}$$

- **Weight initialization**: To initialize $\beta$ to $\beta_0 = 0.85$, $w$ is initialized using the logit function:
  $$w_{\text{init}} = \ln\left(\frac{\beta_0}{1.0 - \beta_0}\right)$$
- **Adaptive Dynamics**: Fast visual layers learn smaller $\beta \approx 0.80$ for rapid transient reaction, while higher decision layers learn larger $\beta \approx 0.90$ to retain working memory across frames.

### 1.3 Non-Differentiable Spike Generation & Surrogate Gradients

When the membrane potential reaches or exceeds the threshold $V_{\text{th}} = 1.0$, the neuron fires a spike:

$$S[t] = \Theta(V[t] - V_{\text{th}}) = \begin{cases} 1, & \text{if } V[t] \ge V_{\text{th}} \\ 0, & \text{otherwise} \end{cases}$$

Because the Heaviside step function $\Theta$ has zero derivative everywhere except at $V = V_{\text{th}}$ (where it is non-differentiable / infinite), standard backpropagation fails.

We resolve this using a **Surrogate Gradient Function** (`snn/spike_function.py`). During the backward pass of Backpropagation Through Time (BPTT), the discontinuous derivative is approximated by a smooth bell-shaped surrogate curve:

$$\frac{\partial S}{\partial V} = \frac{1}{(1 + |V - V_{\text{th}}|)^2}$$

### 1.4 Soft Reset (Reset-by-Subtraction)

After generating a spike, our models use **soft reset** (reset-by-subtraction) rather than hard zeroing:

$$V[t] \leftarrow V[t] - S[t]_{\text{detached}} \cdot V_{\text{th}}$$

**Crucial Implementation Detail**:
By detaching the spike in the reset term ($S[t]_{\text{detached}} = S[t].\text{detach}()$), we prevent catastrophic backward gradient oscillations and numerical instability during BPTT without hindering forward-pass charge retention.

---

## 2. Software Baseline Architecture (84×84)

The baseline model (`models/snn/spiking_cnn.py`: `SpikingDuelingDQN`) processes $84 \times 84$ single-channel game frames across $T=4$ simulation timesteps:

```
Input Frame (4, 84, 84)
       │
       ▼
[Conv2d (4->32, 8x8, s=4) + BatchNorm2d + ParametricLIF]  --> Spikes (32, 20, 20) = 12,800 neurons
       │
       ▼
[Conv2d (32->64, 4x4, s=2) + BatchNorm2d + ParametricLIF] --> Spikes (64, 9, 9)   =  5,184 neurons
       │
       ▼
[Conv2d (64->64, 3x3, s=1) + BatchNorm2d + ParametricLIF] --> Spikes (64, 7, 7)   =  3,136 neurons
       │
       ▼
[Flatten -> Linear(3136->512) + BatchNorm1d + ParametricLIF] --> Spikes (512)     =    512 neurons
       │
  ┌────┴──────────────────────────┐
  ▼                               ▼
[Value Head: Linear(512->1)]   [Advantage Head: Linear(512->5)]
Leaky Readout Integration       Leaky Readout Integration
  └────┬──────────────────────────┘
       ▼
Dueling Q-Value Aggregation:
Q(s, a) = V(s) + (A(s, a) - mean(A(s, :)))
```

Total baseline parameters: **~1,710,000 synaptic weights (FP32)**, **21,632 spiking neurons**.

---

## 3. SNAP-V (Cerebra-H) Hardware-Aware Neuromorphic Architecture

To deploy onto the physical **SNAP-V neuromorphic FPGA accelerator** (RISC-V host + Cerebra-H 32-cluster core), the network must satisfy rigid physical hardware constraints:

1. **Downsampled Spatial Resolution**: Input is $32 \times 32$ pixels (4 stacked frames $\to (4, 32, 32)$).
2. **Fixed Power-of-Two Decay (`ShiftLIF`)**: Zero floating-point multiplications; all decay is performed via hardware bit-shifts.
3. **8-Bit Signed Synaptic Weights (INT8)**: Weights must reside in $[-128, 127]$ to fit FPGA on-chip SRAM.
4. **SRAM Synapse Budget**: Must not exceed SNAP-V's 524,288 weight limit (strictly $< 450,000$).
5. **Network-on-Chip (NoC) Bandwidth**: Spiking activity must stay $< 10\%$ to eliminate packet router congestion.

The hardware-constrained model is implemented in [`models/snn/snapv_snn.py`](file:///home/dilith_s_b_s/UoP/MyProjects/SNN_doomplay/SNN-Plays-Doom/models/snn/snapv_snn.py) (`SnapVSpikingDQN`).

```
Input Frame (4, 32, 32)
       │
       ▼
[QuantConv2d (4->16, 4x4, s=2) + ShiftLIF(k=2, β=0.750)]  --> Spikes (16, 15, 15) = 3,600 neurons
       │
       ▼
[QuantConv2d (16->24, 3x3, s=2) + ShiftLIF(k=3, β=0.875)] --> Spikes (24, 7, 7)   = 1,176 neurons
       │
       ▼
[Flatten (1176) -> QuantLinear(1176->128) + ShiftLIF(k=3, β=0.875)] --> Spikes (128) = 128 neurons
       │
  ┌────┴──────────────────────────┐
  ▼                               ▼
[Value Head: QuantLinear(128->1)] [Advantage Head: QuantLinear(128->5)]
Leaky Readout (k=3, β=0.875)       Leaky Readout (k=3, β=0.875)
  └────┬──────────────────────────┘
       ▼
Dueling Q-Value Output: Q(s, a) (5 Compound Actions)
```

### 3.1 ShiftLIF: Power-of-Two Bit-Shift Decay

FPGA digital hardware implements bit-shifts in a single clock cycle with zero DSP multipliers:

$$V[t] = V[t-1] - (V[t-1] \gg k) + I[t]$$

This matches an exact exponential decay factor $\beta = 1 - 2^{-k}$:
- **$k=2$**: $\beta = 1 - 2^{-2} = 1 - 0.25 = \mathbf{0.750}$ (Used in Conv1 for fast visual transients).
- **$k=3$**: $\beta = 1 - 2^{-3} = 1 - 0.125 = \mathbf{0.875}$ (Used in Conv2, FC1, and Readout for working memory).

Surrogate gradient in PyTorch:
During training on GPU, `ShiftLIF` simulates this exact shift decay, and backpropagates through spikes using the Fast Sigmoid surrogate gradient:

$$\sigma'(v) = \frac{\alpha}{(1 + \alpha |v - V_{\text{th}}|)^2}, \quad \alpha = 4.0$$

### 3.2 Quantization-Aware Training (QAT) with STE

In `QuantConv2d` and `QuantLinear`, weights are quantized to signed 8-bit integers during every forward pass:

$$\text{scale} = \max\left( \frac{\max(|\mathbf{W}|)}{127.0}, 10^{-8} \right)$$
$$\mathbf{W}_{\text{quant}} = \text{clamp}\left( \text{round}\left( \frac{\mathbf{W}}{\text{scale}} \right), -128, 127 \right) \cdot \text{scale}$$

Using the **Straight-Through Estimator (STE)**, the rounding and clamping operations are treated as identity during the backward pass:

$$\mathbf{W}_{\text{ste}} = \mathbf{W} + (\mathbf{W}_{\text{quant}} - \mathbf{W}).\text{detach}()$$

This allows full-precision gradient descent to update the latent weights while the forward pass operates on discrete integer representations.

### 3.3 Hardware Resource Sizing for SNAP-V

| Layer | Kernel / Features | Output Dimensions | Spiking Neurons | Synaptic Weights (INT8) |
|---|---|---|---|---|
| **Conv1** | $16 \times 4 \times 4 \times 4$, stride 2 | $16 \times 15 \times 15$ | 3,600 | 1,024 bytes |
| **Conv2** | $24 \times 16 \times 3 \times 3$, stride 2 | $24 \times 7 \times 7$ | 1,176 | 3,456 bytes |
| **FC1** | $128 \times 1,176$ | 128 | 128 | 150,528 bytes |
| **Value Head** | $1 \times 128$ | 1 (leaky) | 1 | 128 bytes |
| **Advantage Head** | $5 \times 128$ | 5 (leaky) | 5 | 640 bytes |
| **TOTAL** | — | — | **4,910** | **155,776 bytes (152.12 KB)** |

> **SNAP-V Capacity Check**: 155,776 synapses requires **29.71%** of SNAP-V's 524,288 SRAM capacity, passing the $< 450,000$ limit with over 70% headroom remaining for other tasks!

---

## 4. Off-Chip Training & Teacher Distillation

The agent is trained strictly **off-chip on a GPU/CPU workstation** before deployment.

### 4.1 Multi-Objective Loss Formulation

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{TD}} + \lambda_{\text{distill}} \cdot \mathcal{L}_{\text{distill}} + \mathcal{L}_{\text{NoC}}$$

1. **Double DQN (DDQN) Loss**:
   $$y = r + \gamma (1 - d) \cdot Q_{\text{target}}\left(s', \arg\max_{a'} Q_{\text{online}}(s', a')\right)$$
   $$\mathcal{L}_{\text{TD}} = \text{SmoothL1}\big(Q_{\text{online}}(s, a) - y\big)$$

2. **Teacher Knowledge Distillation Loss**:
   The expert baseline SNN ([`checkpoints/snn_policy_best.pth`](file:///home/dilith_s_b_s/UoP/MyProjects/SNN_doomplay/SNN-Plays-Doom/checkpoints/snn_policy_best.pth)) inspects native $84 \times 84$ frames and produces expert Q-value targets $\mathbf{Q}_{\text{teacher}}$:
   $$\mathcal{L}_{\text{distill}} = \text{SmoothL1}\big(\mathbf{Q}_{\text{student}}(s_{32}) - \mathbf{Q}_{\text{teacher}}(s_{84})\big)$$

3. **One-Sided NoC Congestion Barrier**:
   Unlike two-sided penalties that force silent networks to fire unnecessary spikes, our loss acts as a one-sided safety barrier:
   $$\mathcal{L}_{\text{NoC}} = w_{\text{sparse}} \cdot \big(\max(0, \text{FiringRate} - 0.10)\big)^2$$
   Whenever the firing rate is $\le 10\%$, this penalty is identically zero.

### 4.2 5-Action Symmetric Data Augmentation

ViZDoom frames are horizontally flipped with 50% probability during replay sampling. To preserve action semantics across the 5 compound actions:
- `Action 0 (Left)` $\longleftrightarrow$ `Action 1 (Right)`
- `Action 3 (Left + Shoot)` $\longleftrightarrow$ `Action 4 (Right + Shoot)`
- `Action 2 (Shoot)` remains invariant.
- In teacher distillation mode, columns $[0, 1]$ and $[3, 4]$ of $\mathbf{Q}_{\text{teacher}}$ are swapped symmetrically.

---

## 5. Automated Hardware Compiler (`deployment/export_snapv.py`)

The exporter converts PyTorch model weights into formats directly consumable by SNAP-V FPGA tooling:

### 5.1 32-Cluster Hardware Mapping Table

SNAP-V consists of 32 physical neuromorphic clusters, each with dedicated local SRAM:

| Cluster Range | Layer | Allocated Neurons | Fan-In per Neuron | Shift Factor $k$ |
|---|---|---|---|---|
| **Clusters 00 – 19** (20 clusters) | Conv1 | 180 neurons / cluster | 64 | $k=2$ ($\beta=0.750$) |
| **Clusters 20 – 27** ( 8 clusters) | Conv2 | 147 neurons / cluster | 144 | $k=3$ ($\beta=0.875$) |
| **Clusters 28 – 30** ( 3 clusters) | FC1 | 43, 43, 42 neurons | 1,176 | $k=3$ ($\beta=0.875$) |
| **Cluster 31** ( 1 cluster) | Readout | 6 neurons (Value + Adv) | 128 | $k=3$ ($\beta=0.875$) |

### 5.2 Exported Artifacts

1. **[`deployment/snapv_model.bin`](file:///home/dilith_s_b_s/UoP/MyProjects/SNN_doomplay/SNN-Plays-Doom/deployment/snapv_model.bin)** (156 KB):
   - 16-byte header: `0x534E4150` (`"SNAP"`), version `1`, total synapses (`155,776`), total neurons (`4,910`), num layers (`5`).
   - Packed continuous INT8 byte stream for direct DMA transfer into Cerebra-H local SRAM.
2. **[`deployment/snapv_model.h`](file:///home/dilith_s_b_s/UoP/MyProjects/SNN_doomplay/SNN-Plays-Doom/deployment/snapv_model.h)** (980 KB):
   - Static multidimensional arrays: `snapv_conv1_weights[16][4][4][4]`, `snapv_conv2_weights[24][16][3][3]`, `snapv_fc1_weights[128][1176]`, etc.
   - Dequantization scale constants and INT32 biases.
   - 32-cluster mapping configuration struct array `snapv_cluster_map[32]`.
   - RISC-V C reference kernel:
     ```c
     static inline uint8_t snapv_shift_lif_update(int32_t *v_mem, int32_t current, uint8_t shift_k, int32_t threshold) {
         *v_mem = *v_mem - (*v_mem >> shift_k) + current;
         if (*v_mem >= threshold) {
             *v_mem -= threshold;
             return 1;
         }
         return 0;
     }
     ```

---

## 6. Common Pitfalls & Practical Debugging

| Issue / Symptom | Underlying Cause | Correct Solution |
|---|---|---|
| **TD Loss increasing from 0.0 to ~4.0 during RL training** | Normal Q-learning scale expansion: as rewards are discovered, Q-values expand from $\approx 0$ to tens of units, naturally increasing Bellman error magnitudes. | Monitor `Avg50` reward and `Distill` loss (which decreases), not raw TD loss. |
| **Loss is 0.0000 for the first ~50 episodes** | Replay buffer warmup: the model collects transitions until `warmup_steps = 2,000` before triggering gradient updates. | Expected behavior. Training begins automatically once warmup completes. |
| **`argparse --render` parsing errors** | Standard Python `type=bool` treats `"False"` as boolean `True`. | Use `str2bool` helper with both `--render` and `--no-render` flags. |
| **ViZDoom crash: missing X11 display** | Running in a headless terminal or SSH session without GUI forwarding. | Pass `--no-render` or use `xvfb-run -a python evaluation/evaluate_snapv.py`. |
| **Teacher checkpoint not found during training** | Path typo or training baseline was not completed. | Ensure `checkpoints/snn_policy_best.pth` exists or run `python training/train_snn.py` first. |
