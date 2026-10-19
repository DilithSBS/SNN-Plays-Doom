# Developer & Architecture Guide: SNN-Plays-Doom

This guide provides an in-depth technical explanation of the neuromorphic design, mathematical formulations, training mechanics, and codebase architecture in **SNN-Plays-Doom**.

---

## 1. Neuromorphic Fundamentals & SNN Mathematics

Traditional Artificial Neural Networks (ANNs) process information as continuous floating-point activations evaluated in a single forward pass. In contrast, **Spiking Neural Networks (SNNs)** compute temporally using discrete event-based binary spikes ($S \in \{0, 1\}$).

### 1.1 Leaky Integrate-and-Fire (LIF) Dynamics

Every spiking neuron models the electrical properties of a biological cell membrane with a resistor-capacitor (RC) circuit. Over discretized time steps $t \in \{1, \dots, T\}$, incoming synaptic current $I[t]$ charges the membrane potential $V[t]$. Between input pulses, charge leaks through the membrane resistance:

$$V[t] = \beta \cdot V[t-1] + I[t]$$

Where:
- $V[t]$ is the membrane potential at simulation time step $t$.
- $I[t] = \text{BN}(\mathbf{W} \cdot \mathbf{x}[t])$ is the input current produced by synaptic weights and batch normalization.
- $\beta \in (0, 1)$ is the membrane decay constant ($\beta = e^{-\Delta t / \tau_m}$).

### 1.2 Learnable Membrane Decay (Parametric LIF)

Rather than fixing $\beta$ as an arbitrary constant across all layers, our `ParametricLIF` layer learns an optimal decay rate per channel/feature:

$$\beta = \sigma(w) = \frac{1}{1 + e^{-w}}$$

- **Weight initialization**: To initialize $\beta$ to a target value $\beta_0$ (e.g. 0.85), $w$ is initialized using the logit (inverse sigmoid) function:
  $$w_{\text{init}} = \ln\left(\frac{\beta_0}{1.0 - \beta_0}\right)$$
- **Adaptive Dynamics**: Fast visual layers (e.g., Conv1) learn smaller $\beta \approx 0.80$ for rapid transient reaction, while higher association layers learn larger $\beta \approx 0.90$ to retain working memory across frames.

### 1.3 Non-Differentiable Spike Generation & Surrogate Gradients

When the membrane potential reaches or exceeds the threshold $V_{\text{th}} = 1.0$, the neuron fires a spike:

$$S[t] = \Theta(V[t] - V_{\text{th}}) = \begin{cases} 1, & \text{if } V[t] \ge V_{\text{th}} \\ 0, & \text{otherwise} \end{cases}$$

Because the Heaviside step function $\Theta$ has zero derivative everywhere except at $V = V_{\text{th}}$ (where it is non-differentiable / infinite), standard backpropagation fails.

We resolve this using a **Surrogate Gradient Function** (`snn/spike_function.py`). During the backward pass of Backpropagation Through Time (BPTT), the discontinuous derivative is approximated by a smooth bell-shaped surrogate curve:

$$\frac{\partial S}{\partial V} = \frac{1}{(1 + |V - V_{\text{th}}|)^2}$$

This provides well-behaved gradients to update both upstream synaptic weights and learnable membrane decay parameters.

### 1.4 Soft Reset (Reset-by-Subtraction)

After generating a spike, biological neurons reset their membrane potential. Standard SNN implementations frequently use **hard reset** ($V[t] \leftarrow 0$), which discards any excess potential above the threshold.

Our model uses **soft reset** (reset-by-subtraction):

$$V[t] \leftarrow V[t] - S[t]_{\text{detached}} \cdot V_{\text{th}}$$

**Crucial Implementation Detail**:
Notice the detached spike $S[t]_{\text{detached}} = S[t].\text{detach}()$. By detaching the spike in the reset term, we prevent catastrophic backward gradient oscillations and numerical instability during BPTT without hindering forward-pass dynamics.

---

## 2. Spiking Dueling DQN Architecture

The network (`models/snn/spiking_cnn.py`: `SpikingDuelingDQN`) processes $84 \times 84$ single-channel game frames across $T=4$ simulation timesteps.

```
Input Frame (1, 84, 84)
       │
       ▼
[Conv2d (1->32, 8x8, s=4) + BatchNorm2d + ParametricLIF]  --> Spikes (32, 20, 20)
       │
       ▼
[Conv2d (32->64, 4x4, s=2) + BatchNorm2d + ParametricLIF] --> Spikes (64, 9, 9)
       │
       ▼
[Conv2d (64->64, 3x3, s=1) + BatchNorm2d + ParametricLIF] --> Spikes (64, 7, 7)
       │
       ▼
[Flatten (3136) -> Linear(3136->512) + BatchNorm1d + ParametricLIF] --> Spikes (512)
       │
  ┌────┴──────────────────────────┐
  ▼                               ▼
[Value Head: Linear(512->1)]   [Advantage Head: Linear(512->3)]
Leaky Readout Integration       Leaky Readout Integration
Membrane Potential V(s)         Membrane Potential A(s, a)
  └────┬──────────────────────────┘
       ▼
Dueling Q-Value Aggregation:
Q(s, a) = V(s) + (A(s, a) - mean(A(s, :)))
```

### 2.1 Non-Spiking Leaky Readout Heads

Reinforcement learning requires continuous action Q-values $Q(s, a) \in \mathbb{R}$. Rather than counting spikes, our readout heads (`value_head` and `advantage_head`) act as **non-spiking integrator neurons** with decay $\beta_{\text{readout}} = 0.85$:

$$V_{\text{val}}[t] = \beta_{\text{readout}} \cdot V_{\text{val}}[t-1] + \mathbf{W}_v \cdot \mathbf{S}_{\text{fc}}[t]$$
$$A_{\text{adv}}[t] = \beta_{\text{readout}} \cdot A_{\text{adv}}[t-1] + \mathbf{W}_a \cdot \mathbf{S}_{\text{fc}}[t]$$

The final values after $T$ timesteps serve as the state value $V(s)$ and action advantages $A(s, a)$.

### 2.2 Dueling Q-Value Aggregation

$$Q(s, a) = V(s) + \left(A(s, a) - \frac{1}{|\mathcal{A}|} \sum_{a' \in \mathcal{A}} A(s, a')\right)$$

This dueling formulation uncouples state value estimation from action advantages, accelerating value convergence when many actions produce similar outcomes.

---

## 3. Training Pipeline & Loss Formulation

### 3.1 Double DQN (DDQN) Target Calculation

To eliminate maximization bias:
1. The **Online Network** selects the best greedy action for the next state:
   $$a^* = \arg\max_{a'} Q(s_{t+1}, a'; \theta)$$
2. The **Target Network** evaluates the Q-value of that selected action:
   $$y = r + \gamma \cdot (1 - d) \cdot Q(s_{t+1}, a^*; \theta^{-})$$

### 3.2 Homeostatic Firing Rate Regularization

If unconstrained, SNNs can collapse into two failure modes:
1. **Silent Neurons**: Firing rate approaches 0%, stopping gradients.
2. **Epileptic Saturation**: Firing rate approaches 100%, degrading energy efficiency.

We add a differentiable homeostatic sparsity loss:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{TD}}(Q(s, a), y) + \lambda_{\text{sparsity}} \cdot \left(r_{\text{firing}} - r_{\text{target}}\right)^2$$

- $\mathcal{L}_{\text{TD}}$: Huber loss (Smooth L1).
- $r_{\text{target}} = 0.12$ (aims for ~12% overall spike rate).
- $\lambda_{\text{sparsity}} = 0.05$.

### 3.3 Soft Target Update (Polyak Averaging)

Rather than copying target weights periodically, the target network parameters and BatchNorm running buffers are updated softly at every gradient step ($\tau = 0.005$):

$$\theta^{-} \leftarrow (1 - \tau) \cdot \theta^{-} + \tau \cdot \theta$$

---

## 4. Environment & Preprocessing Architecture

### 4.1 Frame Preprocessing (`preprocessing/image_processing.py`)

1. **Format Transposition**: ViZDoom outputs `(C, H, W)` in RGB. OpenCV expects `(H, W, C)`.
2. **Grayscale Conversion**: Eliminates color channels while preserving luminance contrast:
   $$I_{\text{gray}} = 0.299 R + 0.587 G + 0.114 B$$
3. **Downsampling**: Resized to $84 \times 84$ via area interpolation (`cv2.INTER_AREA`).
4. **Normalization**: Pixel values scaled to float32 range $[0.0, 1.0]$.
5. **Output Tensor**: Shape `(1, 84, 84)`.

### 4.2 ViZDoom Action Mapping & Reward Shaping (`environment/doom_env.py`)

- **Scenario**: `configs/basic.cfg` running `configs/basic.wad`.
- **Buttons**:
  - `0`: `MOVE_LEFT`  $\to [1, 0, 0]$
  - `1`: `MOVE_RIGHT` $\to [0, 1, 0]$
  - `2`: `ATTACK`     $\to [0, 0, 1]$
- **Reward Shaping**:
  - ViZDoom grants $+100$ for killing the monster, $-1$ living penalty per tic, and $-300$ for death.
  - **Attack Penalty**: When `ATTACK` is pressed, a $-5.0$ penalty is deducted from the step reward. This prevents the agent from mindlessly holding down fire and forces it to align its crosshairs before shooting.

---

## 5. Development Workflows

### 5.1 Project Layout

```
SNN-Plays-Doom/
├── configs/
│   ├── basic.cfg           # ViZDoom game parameters & key bindings
│   └── basic.wad           # Scenario map asset
├── environment/
│   ├── doom_env.py         # Gymnasium-style wrapper for ViZDoom
│   └── test_vizdoom.py     # ViZDoom sanity test
├── preprocessing/
│   └── image_processing.py # Grayscale conversion, resizing (84x84)
├── models/
│   ├── ann/
│   │   └── cnn.py          # Classical CNN baseline policy
│   └── snn/
│       └── spiking_cnn.py  # SpikingDuelingDQN & ParametricLIF
├── snn/
│   ├── spike_function.py   # PyTorch autograd surrogate gradient
│   ├── lif.py              # Minimal LIF reference
│   └── vizdoom_bridge.py   # Tensor conversion utilities
├── training/
│   ├── replay_buffer.py    # Experience replay buffer
│   ├── train_ann.py        # ANN DQN training loop
│   └── train_snn.py        # SNN Double Dueling DQN training loop
├── evaluation/
│   ├── evaluate_snn.py     # SNN evaluation with live spike telemetry
│   ├── evaluate_ann.py     # ANN evaluation benchmark
│   └── random_baseline.py  # Zero-intelligence baseline
├── checkpoints/            # Pretrained weights (.pth)
├── requirements.txt        # Python dependency manifest
├── README.md               # Main project overview & user guide
└── DEV_GUIDE.md            # This in-depth architecture document
```

### 5.2 Adding a New ViZDoom Scenario

1. Place your scenario `.cfg` and `.wad` files into `configs/` (e.g., `configs/defend_the_center.cfg`).
2. Update the environment call in your training/eval script:
   ```python
   env = DoomEnv(config_path="configs/defend_the_center.cfg", render=False)
   ```
3. Update `num_actions` to match `env.action_space_size`.
4. If the new scenario adds more buttons, adjust your action conversion logic accordingly.

### 5.3 Tuning SNN Simulation Timesteps

In `models/snn/spiking_cnn.py` and `training/train_snn.py`:
- `time_steps = 4` is the default: provides a great balance of temporal dynamics and high FPS (~120 FPS on GPU).
- For deeper temporal integration, increase to `time_steps = 8` or `16`.
- Note: Increasing $T$ linearly scales memory and compute during BPTT.

### 5.4 Live Spike Telemetry (`evaluation/evaluate_snn.py`)

The `SpikeTelemetryMonitor` uses PyTorch forward hooks to inspect firing rates non-intrusively without altering model code:
```python
monitor = SpikeTelemetryMonitor(model)
# ... forward passes ...
stats = monitor.get_step_summary()
# Returns: { 'lif1': {'rate': 0.22, 'spikes': 2816}, ... }
```

---

## 6. Common Pitfalls & Debugging

| Issue | Cause | Fix |
|---|---|---|
| `BatchNorm1d` crash during evaluation | Passing a single observation (`batch_size=1`) while model is in `.train()` mode | Always call `model.eval()` before calling `select_action()` or `forward()`. |
| Zero gradients in SNN | Spike threshold unreachable or membrane potential blown up | Check that input frames are scaled to $[0, 1]$. Ensure `SpikeFunction.apply()` is used instead of standard `(v >= v_th).float()`. |
| ViZDoom window won't open on headless Linux | Missing X11 display server | Pass `--no-render` or run with `xvfb-run -a python evaluation/evaluate_snn.py`. |
| Audio/OpenAL ALSA warnings | System has no active soundcard or running over SSH | These warnings are harmless and can be ignored. ViZDoom continues simulation normally. |
