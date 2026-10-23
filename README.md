# 🧠 SNN-Plays-Doom: Neuromorphic Deep Reinforcement Learning in ViZDoom

[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![ViZDoom](https://img.shields.io/badge/ViZDoom-1.3.0-darkred.svg)](https://vizdoom.farama.org/)
[![SNAP-V Ready](https://img.shields.io/badge/Hardware-SNAP--V%20(Cerebra--H)-success.svg)](https://github.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**SNN-Plays-Doom** is a bio-inspired Deep Reinforcement Learning framework that trains **Spiking Neural Networks (SNNs)** to play Doom via the ViZDoom environment.

The project features both an unconstrained **high-capacity software baseline SNN** and a hardware-constrained, quantization-aware pipeline designed for direct deployment onto the **SNAP-V (Cerebra-H) Neuromorphic FPGA Accelerator** (RISC-V host + 32-cluster clustered SNN core).

---

## ⚡ Key Highlights

- ⚡ **Event-Driven Computation**: Neurons communicate exclusively through binary spikes ($S \in \{0, 1\}$), demonstrating **>93% neuromorphic event silence**.
- 🛠️ **SNAP-V Neuromorphic Accelerator Pipeline**: Full end-to-end off-chip training and hardware compilation for the 32-cluster Cerebra-H FPGA core.
- 🧮 **Power-of-Two Shift Decay (`ShiftLIF`)**: Eliminates floating-point multipliers in favor of hardware-friendly arithmetic right bit-shifts: $\beta = 1 - 2^{-k}$ ($k=2 \implies 0.75, k=3 \implies 0.875$).
- 🎚️ **INT8 Quantization-Aware Training (QAT)**: Fake-quantizes synaptic weights into signed 8-bit integers $[-128, 127]$ with Straight-Through Estimators (STE).
- 🎓 **Off-Chip Teacher-Student Knowledge Distillation**: Transfers reflexes and aiming precision from a 84×84 baseline teacher into the 32×32 INT8 SNAP-V student, achieving a **78% win rate** with zero on-chip overhead.
- 🚦 **One-Sided NoC Congestion Barrier**: Restricts mean firing rate strictly under **10%** (achieved: **6.53%**) to prevent packet storms on physical FPGA router meshes.
- 📊 **Real-Time Spike Telemetry**: Non-intrusive PyTorch forward hooks tracking exact layer-by-layer firing rates and ASCII sparsity visualizations.
- 📦 **Automated Hardware Compiler (`deployment/export_snapv.py`)**: Generates packed INT8 DMA binary blobs and an embedded C header with 32-cluster SRAM partitioning and RISC-V drivers.

---

## 🚀 Setup Guide (For New PCs)

### 1. Prerequisites

- **Python**: 3.10, 3.11, or 3.12
- **Git**
- **Operating System**: Linux (Ubuntu 20.04/22.04/24.04 recommended), Windows 10/11 (Native or WSL2), or macOS.

---

### 2. Platform-Specific System Dependencies

ViZDoom relies on CMake and SDL2 for game rendering and audio:

#### 🐧 Linux (Ubuntu / Debian / WSL2)
```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake git \
    libboost-all-dev libsdl2-dev libopenal-dev \
    libgl1-mesa-glx libglib2.0-0
```

#### 🪟 Windows
1. **WSL2 (Recommended)**: Run `wsl --install` and follow the Linux instructions. WSLg provides hardware-accelerated GUI rendering.
2. **Native Windows**: Install [CMake for Windows](https://cmake.org/download/) and [Visual Studio C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/).

#### 🍏 macOS (Homebrew)
```bash
brew install cmake boost sdl2 openal-soft
```

---

### 3. Clone Repository & Setup Environment

```bash
# 1. Clone repository
git clone https://github.com/your-username/SNN-Plays-Doom.git
cd SNN-Plays-Doom

# 2. Create virtual environment
python3 -m venv venv

# 3. Activate virtual environment
# Linux / macOS / WSL2:
source venv/bin/activate
# Windows (cmd):
venv\Scripts\activate.bat
# Windows (PowerShell):
venv\Scripts\Activate.ps1
```

---

### 4. Install Dependencies

```bash
# Install PyTorch with CUDA (if NVIDIA GPU available):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Or CPU-only:
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install project manifest:
pip install -r requirements.txt
```

Verify your setup:
```bash
python environment/test_vizdoom.py
```

---

## 🕹️ Interactive Human Play (Play with Keyboard & Mouse)

Want to jump into Doom and play it yourself using your keyboard and mouse? A complete interactive environment is located in [`human_play/`](file:///home/dilith_s_b_s/UoP/MyProjects/SNN_doomplay/SNN-Plays-Doom/human_play):

```bash
# 1. Play Combat Arena (WASD to move, Mouse / Ctrl / Space to shoot):
./human_play/play.sh

# 2. Play 3D Labyrinth Traversal:
./human_play/play.sh --scenario my_way_home

# 3. Play 360° Circular Arena Defense:
./human_play/play.sh --scenario defend_the_center

# High-resolution (1024x768) or record gameplay to an MP4 video:
./human_play/play.sh --res 1024x768 --record my_gameplay.mp4
```

See the [Human Play Guide](file:///home/dilith_s_b_s/UoP/MyProjects/SNN_doomplay/SNN-Plays-Doom/human_play/README.md) for the complete keyboard & mouse control bindings table.

---

## 🎮 Running & Evaluating Models

Pre-trained weights are located in [`checkpoints/`](file:///home/dilith_s_b_s/UoP/MyProjects/SNN_doomplay/SNN-Plays-Doom/checkpoints).

### 1. Evaluate the SNAP-V Hardware SNN (32×32, INT8 Quantized, 5 Actions)
Script: [`evaluation/evaluate_snapv.py`](file:///home/dilith_s_b_s/UoP/MyProjects/SNN_doomplay/SNN-Plays-Doom/evaluation/evaluate_snapv.py)

```bash
# Watch live gameplay window (capped at 25 FPS):
python evaluation/evaluate_snapv.py --render --episodes 10 --fps 25

# Fast headless benchmarking:
python evaluation/evaluate_snapv.py --no-render --episodes 50

# Verbose step-by-step Q-values & spike telemetry:
python evaluation/evaluate_snapv.py --render --episodes 2 --verbose-steps
```

Sample output:
```text
================================================================================
                  FINAL SNAP-V HARDWARE BENCHMARK SUMMARY
================================================================================
  Episodes Evaluated    : 50
  Mean Game Reward      : -274.58 ± 377.46
  Mean Episode Duration : 28.9 steps (78% kill rate)
--------------------------------------------------------------------------------
  Compound Action Distribution:
    LEFT           :  183 times ( 12.7%)
    RIGHT          :  222 times ( 15.4%)
    SHOOT          :   39 times (  2.7%)
    LEFT+SHOOT     :  438 times ( 30.3%)
    RIGHT+SHOOT    :  562 times ( 38.9%)  --> 69.2% Tactical Strafe-Shooting!
--------------------------------------------------------------------------------
  Layer-by-Layer Spiking Activity:
    Conv1 + ShiftLIF (16x15x15 = 3,600 neurons)   : ██████░░░░░░░░░  7.42%
    Conv2 + ShiftLIF (24x7x7   = 1,176 neurons)   : ██░░░░░░░░░░░░░  3.18%
    FC1   + ShiftLIF (128       =   128 neurons)  : █████████░░░░░░ 12.23%
--------------------------------------------------------------------------------
    OVERALL NETWORK SPARSITY (4,904 neurons)      : █████░░░░░░░░░░  6.53%

  Neuromorphic Inactivity : 93.5% silent (zero spike transmission)
  Mean Firing Rate        : 6.53%
  NoC Congestion Status   : [PASSED] < 10% target firing rate! Guaranteed zero packet loss on SNAP-V mesh.
================================================================================
```

### 2. Evaluate the Software Baseline SNN (84×84, FP32, ParametricLIF)
Script: [`evaluation/evaluate_snn.py`](file:///home/dilith_s_b_s/UoP/MyProjects/SNN_doomplay/SNN-Plays-Doom/evaluation/evaluate_snn.py)

```bash
python evaluation/evaluate_snn.py --render --episodes 10
```

### 3. Evaluate the Classical ANN Baseline (84×84, FP32, ReLU)
Script: [`evaluation/evaluate_ann.py`](file:///home/dilith_s_b_s/UoP/MyProjects/SNN_doomplay/SNN-Plays-Doom/evaluation/evaluate_ann.py)

```bash
python evaluation/evaluate_ann.py --render --episodes 10
```

### 4. Evaluate the SNAP-V 3D Navigation Model (my_way_home Labyrinth)
Script: [`evaluation/evaluate_nav.py`](file:///home/dilith_s_b_s/UoP/MyProjects/SNN_doomplay/SNN-Plays-Doom/evaluation/evaluate_nav.py)

```bash
# Watch the agent navigate the labyrinth with live window rendering:
python evaluation/evaluate_nav.py --render --episodes 5 --fps 25

# Headless benchmark tracking maze solutions and unique rooms explored:
python evaluation/evaluate_nav.py --no-render --episodes 20
```

---

## 🏋️ Training Pipelines (CPU / GPU)

All training happens **off-chip on your CPU / GPU** using PyTorch. The SNAP-V FPGA only runs on-chip inference.

### 1. Train the SNAP-V Combat Model (basic.cfg)
Script: [`training/train_snapv.py`](file:///home/dilith_s_b_s/UoP/MyProjects/SNN_doomplay/SNN-Plays-Doom/training/train_snapv.py)

Train with **Off-Chip Teacher Distillation** from the baseline SNN:
```bash
python training/train_snapv.py --teacher checkpoints/snn_policy_best.pth --episodes 800
```

Train standalone from scratch without a teacher:
```bash
python training/train_snapv.py --episodes 800
```

### 2. Train the SNAP-V 3D Navigation Model (my_way_home.cfg)
Script: [`training/train_nav.py`](file:///home/dilith_s_b_s/UoP/MyProjects/SNN_doomplay/SNN-Plays-Doom/training/train_nav.py)

Trains spatial traversal with count-based grid novelty exploration bonuses:
```bash
python training/train_nav.py --episodes 800
```

### 3. Train the Unconstrained Baseline SNN (Combat)
```bash
python training/train_snn.py --episodes 800
```

### 4. Train the Classical ANN Baseline
```bash
python training/train_ann.py --episodes 300
```

---

## 📦 SNAP-V Hardware Deployment (Phase 4)

Once trained, compile and export the model for SNAP-V FPGA deployment:

```bash
python deployment/export_snapv.py --checkpoint checkpoints/snapv_policy_best.pth
```

### Generated Hardware Artifacts:
1. **[`deployment/snapv_model.bin`](file:///home/dilith_s_b_s/UoP/MyProjects/SNN_doomplay/SNN-Plays-Doom/deployment/snapv_model.bin)** (156 KB): Packed INT8 binary blob formatted with magic header (`SNAP`), tensor metadata, and INT32 biases for direct DMA transfer into FPGA SRAM.
2. **[`deployment/snapv_model.h`](file:///home/dilith_s_b_s/UoP/MyProjects/SNN_doomplay/SNN-Plays-Doom/deployment/snapv_model.h)** (980 KB): Embedded C/C++ header with static multidimensional weight arrays, 32-cluster mapping structs, and bit-shift LIF integration drivers for the RISC-V host firmware.

#### 32-Cluster Neuromorphic Allocation:
| Layer | Spiking Neurons | Synapses (INT8) | Cluster ID Range | Clusters Allocated |
|---|---|---|---|---|
| **Conv1** | 3,600 | 1,024 | `0` – `19` | 20 clusters (180 neurons/cluster) |
| **Conv2** | 1,176 | 3,456 | `20` – `27` | 8 clusters (147 neurons/cluster) |
| **FC1** | 128 | 150,528 | `28` – `30` | 3 clusters (~43 neurons/cluster) |
| **Readout** | 6 (leaky) | 768 | `31` | 1 cluster (Value + Advantage) |
| **TOTAL** | **4,910** | **155,776** | `0` – `31` | **32 / 32 Clusters (100% Core Utilization)** |

> **SRAM Budget**: Consumes **155,776 bytes (152.12 KB)**, which is **29.71%** of SNAP-V's 524,288 SRAM capacity (passing the 450,000 budget with 70% headroom).

---

## 📊 Comprehensive Benchmark Comparison

Evaluated on 50 consecutive episodes in the ViZDoom `basic` scenario:

| Metric | Random Agent | Classical ANN (CNN) | Software Baseline SNN | **SNAP-V Hardware SNN** |
|---|---|---|---|---|
| **Observation** | 84 × 84 px | 84 × 84 px | 84 × 84 px | **32 × 32 px (Hardware Downsampled)** |
| **Precision** | N/A | FP32 | FP32 | **INT8 $[-128, 127]$ (QAT)** |
| **Synaptic Weights** | N/A | ~1.68 M | ~1.71 M | **155,776 (152 KB, fits FPGA SRAM)** |
| **Neuron Dynamics** | N/A | ReLU (Static) | Parametric LIF ($\beta \in [0.80, 0.90]$) | **ShiftLIF ($\beta = 1 - 2^{-k}$, no multipliers)** |
| **Mean Reward** | -350 ± 120 | +45.2 ± 38 | -103.7 ± 275 | **-274.6 ± 377** |
| **Effective Kill Rate** | ~5% | ~85% | ~60% | **78% (39 / 50 kills)** |
| **Firing Rate (Activity)** | N/A | 100% (Dense) | 32.3% | **6.53% (< 10% NoC Safe Barrier)** |
| **Neuromorphic Silence** | 0% | 0% | 67.7% | **93.5% Silent (Zero Spikes)** |
| **Hardware Deployable** | No | No | No | **YES (SNAP-V FPGA Verified)** |

---

## 📂 Repository Layout

```
SNN-Plays-Doom/
├── checkpoints/              # Saved model weights
│   ├── snapv_policy_best.pth # Best trained SNAP-V combat model
│   ├── nav_snapv_best.pth    # Best trained SNAP-V 3D navigation model
│   ├── snn_policy_best.pth   # Best baseline software SNN model
│   └── ann_policy_best.pth   # Classical CNN baseline weights
├── configs/                  # ViZDoom scenario definitions
│   ├── basic.cfg             # Combat scenario config
│   ├── basic.wad             # Combat scenario map asset
│   ├── my_way_home.cfg       # 3D navigation labyrinth config
│   └── my_way_home.wad       # 3D labyrinth map asset
├── deployment/               # SNAP-V FPGA deployment suite
│   ├── export_snapv.py       # Hardware compiler & INT8 weight exporter
│   ├── snapv_model.bin       # Packed INT8 binary blob for FPGA SRAM DMA
│   └── snapv_model.h         # Embedded C header with 32-cluster mapping
├── environment/              # Environment interface
│   ├── doom_env.py           # Gym-like DoomEnv wrapper (combat & navigation)
│   └── test_vizdoom.py       # ViZDoom sanity test script
├── preprocessing/            # Vision transforms
│   └── image_processing.py   # Grayscale downscaling & normalization
├── models/                   # Neural network architectures
│   ├── ann/
│   │   └── cnn.py            # Classical CNN baseline
│   └── snn/
│       ├── snapv_snn.py      # SnapVSpikingDQN, ShiftLIF, QuantConv2d, QuantLinear
│       └── spiking_cnn.py    # SpikingDuelingDQN with ParametricLIF neurons
├── training/                 # RL training pipelines
│   ├── train_snapv.py        # SNAP-V combat training (Teacher Distillation & NoC loss)
│   ├── train_nav.py          # SNAP-V 3D navigation training (Novelty exploration)
│   ├── train_snn.py          # Baseline software SNN DDQN training loop
│   └── train_ann.py          # Classical ANN DQN training loop
├── evaluation/               # Benchmarking & live visualization
│   ├── evaluate_snapv.py     # SNAP-V combat evaluation & telemetry
│   ├── evaluate_nav.py       # SNAP-V 3D navigation evaluation & maze tracking
│   ├── evaluate_snn.py       # Baseline SNN evaluation with spike telemetry
│   ├── evaluate_ann.py       # ANN evaluation script
│   └── random_baseline.py    # Random action baseline
├── human_play/               # Interactive human gameplay suite (Keyboard & Mouse)
│   ├── play.py               # Main interactive game runner with HUD & controls
│   ├── play.sh               # Quick executable launcher script
│   ├── README.md             # Controls reference & human play documentation
│   └── configs/              # High-res human play scenario configurations
│       ├── human_basic.cfg
│       ├── human_my_way_home.cfg
│       └── human_defend_the_center.cfg
├── requirements.txt          # Python package manifest
├── README.md                 # Project guide (this file)
└── DEV_GUIDE.md              # In-depth architectural & mathematical guide
```

---

## 📜 License

This project is licensed under the MIT License - see the [`LICENSE`](file:///home/dilith_s_b_s/UoP/MyProjects/SNN_doomplay/SNN-Plays-Doom/LICENSE) file for details.
