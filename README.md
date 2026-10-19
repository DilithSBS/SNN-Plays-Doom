# 🧠 SNN-Plays-Doom: Neuromorphic Deep Reinforcement Learning in ViZDoom

[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![ViZDoom](https://img.shields.io/badge/ViZDoom-1.3.0-darkred.svg)](https://vizdoom.farama.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**SNN-Plays-Doom** is a bio-inspired Deep Reinforcement Learning framework that trains **Spiking Neural Networks (SNNs)** to play Doom via the ViZDoom environment.

Using **Leaky Integrate-and-Fire (LIF)** neurons, **learnable membrane dynamics**, **surrogate gradient descent**, and a **Spiking Dueling Deep Q-Network (DQN)** architecture, the agent learns visual combat policies directly from raw pixel frames. It achieves competitive gameplay while demonstrating **neuromorphic event sparsity (~77% silent neurons)** compared to traditional dense Artificial Neural Networks (ANNs).

---

## ⚡ Highlights

- ⚡ **Event-Driven Computation**: Neurons communicate exclusively through binary spikes ($S \in \{0, 1\}$), mimicking biological cortex efficiency.
- 🧬 **Parametric LIF Neurons (`PLIF`)**: Per-channel learnable membrane decay constants ($\beta = \sigma(w)$) bounded in $(0, 1)$.
- 🔁 **Soft Resets with Detached Spikes**: Preserves excess membrane potential above threshold without gradient oscillations during Backpropagation Through Time (BPTT).
- 🎮 **Spiking Dueling DQN**: Dueling streams for state value $V(s)$ and advantage $A(s, a)$ with non-spiking leaky membrane integrator readouts.
- 📊 **Real-Time Spike Telemetry**: Non-intrusive forward hooks tracking exact layer-by-layer firing rates and ASCII sparsity visualizations.
- ⚖️ **ANN vs. SNN Fair Benchmark**: Includes a classical CNN DQN baseline trained under identical game dynamics.

---

## 🚀 Setup Guide (For New PCs)

Follow these step-by-step instructions to get the project cloned, installed, and running on any PC.

### 1. Prerequisites

- **Python**: 3.10, 3.11, or 3.12
- **Git**
- **Operating System**: Linux (Ubuntu 20.04/22.04/24.04 recommended), Windows 10/11 (Native or WSL2), or macOS (Apple Silicon / Intel).

---

### 2. Platform-Specific System Dependencies

ViZDoom relies on CMake and SDL2 for game rendering and audio. Install the prerequisites for your OS:

#### 🐧 Linux (Ubuntu / Debian)
```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake git \
    libboost-all-dev libsdl2-dev libopenal-dev \
    libgl1-mesa-glx libglib2.0-0
```

#### 🪟 Windows
1. **Option A (WSL2 - Strongly Recommended)**:
   Install Ubuntu in WSL2 (`wsl --install`) and follow the Linux instructions above. WSLg provides out-of-the-box hardware-accelerated GUI rendering.
2. **Option B (Native Windows)**:
   - Install [CMake for Windows](https://cmake.org/download/) and add it to your PATH.
   - Install [Visual Studio C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/).
   - OpenAL and SDL2 are bundled with pre-built `vizdoom` wheels on Python 3.10-3.12.

#### 🍏 macOS (Homebrew)
```bash
brew install cmake boost sdl2 openal-soft
```

---

### 3. Clone Repository & Setup Environment

Open your terminal or command prompt:

```bash
# 1. Clone the repository
git clone https://github.com/your-username/SNN-Plays-Doom.git
cd SNN-Plays-Doom

# 2. Create a virtual environment
python3 -m venv venv

# 3. Activate the virtual environment
# On Linux / macOS / WSL2:
source venv/bin/activate
# On Windows (cmd):
venv\Scripts\activate.bat
# On Windows (PowerShell):
venv\Scripts\Activate.ps1
```

*(Alternatively, you can use Conda: `conda create -n snn-doom python=3.12 -y && conda activate snn-doom`)*

---

### 4. Install PyTorch

Install the appropriate PyTorch build for your hardware:

#### With NVIDIA GPU (CUDA 12.1 / 12.4):
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

#### CPU Only (Laptops / Systems without dedicated NVIDIA GPU):
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

---

### 5. Install Project Dependencies

Install the remaining dependencies using the included [`requirements.txt`](file:///home/dilith_s_b_s/UoP/MyProjects/SNN_doomplay/SNN-Plays-Doom/requirements.txt):

```bash
pip install -r requirements.txt
```

---

### 6. Verify Installation

Run the ViZDoom environment test to confirm everything is working:

```bash
python environment/test_vizdoom.py
```

If successful, you will see ViZDoom initialize, step through 10 frames, and close cleanly.

---

## 🎮 Running the Models (Quick Tour)

Pre-trained weights for both the SNN and ANN agents are included in the [`checkpoints/`](file:///home/dilith_s_b_s/UoP/MyProjects/SNN_doomplay/SNN-Plays-Doom/checkpoints) folder.

### 1. Watch the Pre-Trained SNN Agent Play (with Live Telemetry)

Launch the trained Spiking Dueling DQN agent with interactive rendering:

```bash
# Watch with live gameplay window (25 FPS playback)
python evaluation/evaluate_snn.py --episodes 5

# Headless / Fast benchmark without GUI window:
python evaluation/evaluate_snn.py --no-render --episodes 10

# Per-step detailed firing rate logging:
python evaluation/evaluate_snn.py --no-render --episodes 2 --verbose-steps
```

Sample output telemetry:
```text
==============================================================================
                SNN VIZDOOM EVALUATION & SPIKE TELEMETRY REPORT
==============================================================================
  Episodes Evaluated : 10
  Mean Game Reward   : -260.00 ± 45.00
  Mean Survival Steps: 300.0 steps
------------------------------------------------------------------------------
  Layer-by-Layer Spiking Activity (Mean across all episodes & timesteps):
  Conv1 + PLIF (32x20x20 = 12,800 neurons)    : █████████░░░░░░ 22.96%
  Conv2 + PLIF (64x9x9   =  5,184 neurons)    : █████████░░░░░░ 22.72%
  Conv3 + PLIF (64x7x7   =  3,136 neurons)    : █████████░░░░░░ 23.56%
  FC1   + PLIF (512      =    512 neurons)    : ██████████░░░░░ 27.96%
------------------------------------------------------------------------------
  OVERALL NETWORK SPARSITY (21,632 neurons)   : █████████░░░░░░ 23.11%

  Neuromorphic Event Sparsity: 76.9% of neuron-timesteps are silent (0 spikes).
==============================================================================
```

### 2. Evaluate the Classical ANN (CNN) Baseline

```bash
# Watch the ANN agent play with GUI window:
python evaluation/evaluate_ann.py --episodes 5

# Fast headless evaluation:
python evaluation/evaluate_ann.py --no-render --episodes 10
```

### 3. Run the Random Agent Baseline

```bash
python evaluation/random_baseline.py --no-render --episodes 10
```

---

## 🏋️ Training from Scratch

Both training scripts are self-contained, automatically detect CUDA/CPU, and save checkpoints to `checkpoints/`.

### Train the Spiking Dueling DQN
```bash
python training/train_snn.py
```
- Trains for 800 episodes using Double DQN (DDQN) with homeostatic firing rate regularization.
- Saves the best policy to `checkpoints/snn_policy_best.pth` and final model to `checkpoints/snn_policy_final.pth`.

### Train the Classical ANN Baseline
```bash
python training/train_ann.py
```
- Trains for 300 episodes using standard DQN with target network synchronization.
- Saves checkpoints to `checkpoints/ann_policy_best.pth`.

---

## 📂 Repository Structure

```
SNN-Plays-Doom/
├── checkpoints/              # Saved model weights
│   ├── snn_policy_best.pth   # Best trained SNN model
│   ├── snn_policy_final.pth  # Final episode SNN weights
│   └── ann_policy_best.pth   # Classical ANN CNN weights
├── configs/                  # ViZDoom scenario definitions
│   ├── basic.cfg             # Game config (3 actions, 320x240 resolution)
│   └── basic.wad             # Doom map binary
├── environment/              # Environment interface
│   ├── doom_env.py           # Gym-like DoomEnv wrapper with reward shaping
│   └── test_vizdoom.py       # ViZDoom sanity test script
├── preprocessing/            # Vision transforms
│   └── image_processing.py   # Grayscale downscaling (84x84) & normalization
├── models/                   # Neural network architectures
│   ├── ann/
│   │   └── cnn.py            # Classical CNN baseline
│   └── snn/
│       └── spiking_cnn.py    # SpikingDuelingDQN with ParametricLIF neurons
├── snn/                      # Core neuromorphic primitives
│   ├── spike_function.py     # Autograd Surrogate Gradient (Fast Sigmoid)
│   ├── lif.py                # Single-neuron LIF reference implementation
│   ├── models.py             # Linear SNN models for synthetic tasks
│   └── vizdoom_bridge.py     # PyTorch tensor bridge for ViZDoom frames
├── training/                 # RL training pipelines
│   ├── train_snn.py          # SNN Double Dueling DQN training loop
│   ├── train_ann.py          # Classical ANN DQN training loop
│   └── replay_buffer.py      # Experience replay memory
├── evaluation/               # Benchmarking & live visualization
│   ├── evaluate_snn.py       # SNN evaluation & layerwise spike telemetry
│   ├── evaluate_ann.py       # ANN evaluation script
│   └── random_baseline.py    # Random action baseline
├── requirements.txt          # Python package manifest
├── README.md                 # Project guide (this file)
└── DEV_GUIDE.md              # In-depth architectural & mathematical guide
```

---

## 📊 Benchmark & Performance Comparison

Evaluated on 10 consecutive episodes in the `basic` scenario:

| Metric | Random Agent | Classical ANN (CNN) | Spiking Dueling DQN (SNN) |
|---|:---:|:---:|:---:|
| **Mean Reward** | $-905.0 \pm 0.0$ | $-300.0 \pm 0.0$ | **$-260.0 \pm 45.0$** |
| **Survival Steps** | $300$ (Max timeout) | $300$ (Max timeout) | $300$ (Max timeout) |
| **Computation Mode** | N/A | Dense Floating-Point | **Sparse Binary Spikes** |
| **Neuron Sparsity** | N/A | $0.0\%$ (All active) | **$76.9\%$ (Silent)** |
| **Average Firing Rate** | N/A | Continuous activations | **$23.1\%$** |

---

## 📘 Developer Guide & Extending the Project

Looking to dive deeper into the mathematics, customize network dynamics, or add new Doom scenarios?

👉 **Read the comprehensive [Developer & Architecture Guide (`DEV_GUIDE.md`)](file:///home/dilith_s_b_s/UoP/MyProjects/SNN_doomplay/SNN-Plays-Doom/DEV_GUIDE.md)**

Key topics covered in `DEV_GUIDE.md`:
1. **Mathematical Formulations**: LIF differential equations, learnable decay parameterization, soft reset vs. hard reset, and surrogate gradient backward derivation.
2. **Network Internals**: Dueling architecture layout, non-spiking leaky readout heads, and Polyak target updates.
3. **Loss Functions**: Huber loss combined with homeostatic firing rate regularization.
4. **Custom Scenarios**: How to plug in `defend_the_center`, `deadly_corridor`, and add multi-action spaces.
5. **Hyperparameter Tuning**: Adjusting time steps $T$, firing rate targets, and learning rates.

---

## 🛠️ Troubleshooting & FAQs

### 1. `ImportError: libGL.so.1: cannot open shared object file`
Occurs on minimal Linux systems missing OpenGL:
```bash
sudo apt-get install -y libgl1-mesa-glx libglib2.0-0
```

### 2. ViZDoom window fails to open (`DISPLAY not found` / SSH / Headless)
If running over SSH or in a headless Docker/remote server:
- Always pass `--no-render` to evaluation scripts:
  ```bash
  python evaluation/evaluate_snn.py --no-render
  ```
- Or use `xvfb` to create a virtual display:
  ```bash
  sudo apt-get install -y xvfb
  xvfb-run -a python evaluation/evaluate_snn.py
  ```

### 3. ALSA / OpenAL audio warnings
You may see warnings like:
```text
ALSA lib ...: No such device or address
AL lib: ...: Failed to initialize backend "alsa"
```
These warnings are completely harmless and occur when your system doesn't have an active audio output device. ViZDoom handles audio silently in the background.

### 4. Running out of GPU memory
The SNN runs efficiently in under **1.5 GB VRAM**. If you encounter VRAM constraints:
- In `training/train_snn.py`, reduce `batch_size` from `64` to `32`.
- Reduce simulation timesteps `time_steps` from `4` to `2`.

---

## 📜 License

This project is licensed under the MIT License - see the [`LICENSE`](file:///home/dilith_s_b_s/UoP/MyProjects/SNN_doomplay/SNN-Plays-Doom/LICENSE) file for details.
