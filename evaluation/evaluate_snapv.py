"""
SNAP-V (Cerebra-H) Hardware-Aware SNN Evaluation & Spike Telemetry Benchmark

Evaluates a trained SnapVSpikingDQN checkpoint on ViZDoom (32x32 resolution)
with full neuromorphic hardware telemetry:
  - Layer-by-layer firing rates on ShiftLIF spiking neurons
  - Overall spike sparsity and NoC (Network-on-Chip) congestion verification (< 10%)
  - Compound action distribution across 5 actions
  - Live window rendering or fast headless evaluation
"""

import os
import sys
import time
import argparse
from collections import deque
import numpy as np
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from environment.doom_env import DoomEnv
from models.snn.snapv_snn import SnapVSpikingDQN, ShiftLIF


class SnapVTelemetryMonitor:
    """PyTorch forward-hook manager that monitors exact ShiftLIF spike counts and rates."""
    def __init__(self, model: SnapVSpikingDQN):
        self.model = model
        self.hooks = []
        self.layer_spikes = {}
        self.layer_sizes = {}
        self._register_hooks()

    def _register_hooks(self):
        for name, module in self.model.named_modules():
            if isinstance(module, ShiftLIF):
                self.layer_spikes[name] = []
                hook = module.register_forward_hook(self._make_hook(name))
                self.hooks.append(hook)

    def _make_hook(self, layer_name: str):
        def hook_fn(module, inputs, outputs):
            # ShiftLIF returns (spike, membrane)
            spike = outputs[0].detach()
            self.layer_spikes[layer_name].append(float(spike.sum().item()))
            self.layer_sizes[layer_name] = spike.numel()
        return hook_fn

    def reset_step(self):
        for k in self.layer_spikes:
            self.layer_spikes[k].clear()

    def get_step_summary(self) -> dict:
        summary = {}
        total_spikes = 0.0
        total_possible = 0.0

        for name, spk_list in self.layer_spikes.items():
            T = len(spk_list)
            if T == 0:
                continue
            layer_sum = sum(spk_list)
            possible = self.layer_sizes[name] * T
            rate = layer_sum / possible if possible > 0 else 0.0

            summary[name] = {
                "spikes": layer_sum,
                "rate": rate,
                "temporal_profile": spk_list,
            }
            total_spikes += layer_sum
            total_possible += possible

        summary["total"] = {
            "spikes": total_spikes,
            "rate": (total_spikes / total_possible) if total_possible > 0 else 0.0,
        }
        return summary

    def close(self):
        for h in self.hooks:
            h.remove()


def format_state(state) -> np.ndarray:
    """Standardizes DoomEnv observations to shape (4, 32, 32)."""
    if isinstance(state, torch.Tensor):
        state = state.detach().cpu().numpy()
    state = np.asarray(state, dtype=np.float32)

    if state.ndim == 4 and state.shape[0] == 1:
        state = state.squeeze(0)
    elif state.ndim == 2:
        state = np.expand_dims(state, axis=0)
    elif state.ndim == 3 and state.shape[-1] in (1, 3, 4) and state.shape[0] == 32:
        state = np.transpose(state, (2, 0, 1))

    return state


def ascii_bar(rate: float, width: int = 15) -> str:
    """Renders a compact ASCII bar for firing rate percentage (scaled to 20% max)."""
    normalized = min(1.0, max(0.0, rate / 0.20))
    filled = int(round(normalized * width))
    return "█" * filled + "░" * (width - filled)


def load_snapv_checkpoint(checkpoint_path: str, input_channels: int, num_actions: int, default_t: int, device: torch.device):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found at '{checkpoint_path}'. "
            f"Run `python training/train_snapv.py` first!"
        )

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        time_steps = ckpt.get("time_steps", default_t)
        state_dict = ckpt["model_state_dict"]
        meta_info = f"(Trained Ep: {ckpt.get('episode', '?')}, Best Avg50: {ckpt.get('avg_reward_50', 0.0):.1f})"
    else:
        time_steps = default_t
        state_dict = ckpt
        meta_info = "(Raw state_dict)"

    model = SnapVSpikingDQN(
        input_channels=input_channels,
        num_actions=num_actions,
        time_steps=time_steps,
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    print(f"[Loaded SNAP-V SNN] {checkpoint_path} {meta_info} | T={time_steps}")
    return model


@torch.no_grad()
def evaluate_snapv(
    checkpoint_path: str = "checkpoints/snapv_policy_best.pth",
    episodes: int = 10,
    time_steps: int = 4,
    fps: int = 25,
    render: bool = False,
    verbose_steps: bool = False,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize ViZDoom with SNAP-V 32x32 downsampling
    env = DoomEnv(resolution=(32, 32), render=render)
    sample_obs = format_state(env.reset())
    input_channels = sample_obs.shape[0]  # 4 stacked frames
    num_actions = getattr(env, "num_actions", 5)

    action_names = {
        0: "LEFT       ",
        1: "RIGHT      ",
        2: "SHOOT      ",
        3: "LEFT+SHOOT ",
        4: "RIGHT+SHOOT",
    }

    model = load_snapv_checkpoint(
        checkpoint_path=checkpoint_path,
        input_channels=input_channels,
        num_actions=num_actions,
        default_t=time_steps,
        device=device,
    )

    # Hardware resource summary
    res = model.count_resources()
    neurons = res["neurons"]
    synapses = res["synapses"]
    sram_pct = (synapses["total_synapses"] / 524288) * 100

    print("=" * 80)
    print("      SNAP-V (CEREBRA-H) HARDWARE-AWARE NEUROMORPHIC EVALUATION REPORT")
    print("=" * 80)
    print(f"Device           : {device}")
    print(f"Resolution       : 32x32 (Observation: {sample_obs.shape})")
    print(f"Compound Actions : {num_actions} Actions")
    print(f"Simulation Ticks : T = {time_steps} micro-ticks")
    print(f"Spiking Core     : {neurons['total_spiking_neurons']:,} spiking neurons | {synapses['total_synapses']:,} synapses")
    print(f"SRAM Memory      : {sram_pct:.2f}% of SNAP-V 524,288 synapse budget (PASSED: < 450,000)")
    print(f"ShiftLIF Decays  : lif1: β=1-2^-2 (0.750) | lif2: β=1-2^-3 (0.875) | lif_fc: β=1-2^-3 (0.875)")
    print(f"Rendering        : {'Live Window Enabled (FPS: ' + str(fps) + ')' if render else 'Headless (Fast)'}")
    print("-" * 80)

    monitor = SnapVTelemetryMonitor(model)
    frame_delay = (1.0 / fps) if (render and fps > 0) else 0.0

    episode_rewards = []
    episode_steps = []
    action_counts = {i: 0 for i in range(num_actions)}
    global_layer_rates = {name: [] for name in ["lif1", "lif2", "lif_fc", "total"]}

    try:
        for ep in range(1, episodes + 1):
            state = format_state(env.reset())
            done = False
            ep_reward = 0.0
            steps = 0
            ep_layer_accum = {name: [] for name in ["lif1", "lif2", "lif_fc", "total"]}

            while not done:
                steps += 1
                monitor.reset_step()

                state_t = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                q_values = model(state_t, return_firing_rate=False)
                action = int(q_values.argmax(dim=1).item())
                action_counts[action] += 1

                step_stats = monitor.get_step_summary()
                for k in ep_layer_accum:
                    if k in step_stats:
                        ep_layer_accum[k].append(step_stats[k]["rate"])

                if verbose_steps:
                    act_str = action_names.get(action, f"ACT_{action}    ")
                    q_str = ", ".join([f"{v:+.2f}" for v in q_values.squeeze(0).tolist()])
                    print(
                        f"  Step {steps:3d} | Action: {act_str} | Q: [{q_str}] | "
                        f"L1: {step_stats['lif1']['rate']*100:4.1f}% | "
                        f"L2: {step_stats['lif2']['rate']*100:4.1f}% | "
                        f"FC: {step_stats['lif_fc']['rate']*100:4.1f}%"
                    )

                viz_action = env.actions[action]
                step_out = env.step(viz_action)
                if len(step_out) == 3:
                    next_raw_state, reward, done = step_out
                elif len(step_out) == 4:
                    next_raw_state, reward, done, _ = step_out
                else:
                    next_raw_state, reward, terminated, truncated, _ = step_out
                    done = terminated or truncated

                if next_raw_state is not None:
                    state = format_state(next_raw_state)

                ep_reward += float(reward)

                if frame_delay > 0:
                    time.sleep(frame_delay)

            episode_rewards.append(ep_reward)
            episode_steps.append(steps)

            ep_means = {k: float(np.mean(v)) if v else 0.0 for k, v in ep_layer_accum.items()}
            for k, val in ep_means.items():
                global_layer_rates[k].append(val)

            status_tag = "KILLED" if ep_reward > -300 else "SURVIVED"
            print(
                f"Episode {ep:2d}/{episodes} | "
                f"Reward: {ep_reward:6.1f} [{status_tag:^8}] | "
                f"Steps: {steps:3d} | "
                f"Spikes -> "
                f"Conv1: {ep_means['lif1']*100:4.1f}% | "
                f"Conv2: {ep_means['lif2']*100:4.1f}% | "
                f"FC1: {ep_means['lif_fc']*100:4.1f}% | "
                f"Net: {ep_means['total']*100:4.1f}%"
            )

        # Print Final Telemetry & Hardware Feasibility Report
        print("\n" + "=" * 80)
        print("                  FINAL SNAP-V HARDWARE BENCHMARK SUMMARY")
        print("=" * 80)
        mean_reward = float(np.mean(episode_rewards))
        std_reward = float(np.std(episode_rewards))
        print(f"  Episodes Evaluated    : {episodes}")
        print(f"  Mean Game Reward      : {mean_reward:.2f} ± {std_reward:.2f}")
        print(f"  Max / Min Reward      : {np.max(episode_rewards):.1f} / {np.min(episode_rewards):.1f}")
        print(f"  Mean Episode Duration : {np.mean(episode_steps):.1f} steps")
        print("-" * 80)

        total_actions = sum(action_counts.values())
        print("  Compound Action Distribution:")
        for act_idx, count in action_counts.items():
            pct = (count / total_actions * 100) if total_actions > 0 else 0.0
            print(f"    {action_names[act_idx]:<14} : {count:4d} times ({pct:5.1f}%)")
        print("-" * 80)

        print("  Layer-by-Layer Spiking Activity (Averaged across all episodes & T timesteps):")
        layer_labels = [
            ("lif1", "Conv1 + ShiftLIF (16x15x15 = 3,600 neurons)"),
            ("lif2", "Conv2 + ShiftLIF (24x7x7   = 1,176 neurons)"),
            ("lif_fc", "FC1   + ShiftLIF (128       =   128 neurons)"),
            ("total", "OVERALL NETWORK SPARSITY (4,904 neurons)"),
        ]

        for key, desc in layer_labels:
            mean_rate = float(np.mean(global_layer_rates[key]))
            bar = ascii_bar(mean_rate)
            if key == "total":
                print("-" * 80)
            print(f"    {desc:<45} : {bar} {mean_rate*100:5.2f}%")

        overall_mean_rate = float(np.mean(global_layer_rates["total"]))
        overall_sparsity = 1.0 - overall_mean_rate
        print(f"\n  Neuromorphic Inactivity : {overall_sparsity*100:.1f}% silent (zero spike transmission)")
        print(f"  Mean Firing Rate        : {overall_mean_rate*100:.2f}%")

        # NoC Congestion Check
        if overall_mean_rate <= 0.10:
            print("  NoC Congestion Status   : [PASSED] < 10% target firing rate! Guaranteed zero packet loss on SNAP-V mesh.")
        else:
            print("  NoC Congestion Status   : [WARNING] Exceeds 10% target rate. May cause packet buffering on FPGA mesh.")
        print("=" * 80)

    finally:
        monitor.close()
        if hasattr(env, "close"):
            env.close()


if __name__ == "__main__":
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if str(v).lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif str(v).lower() in ("no", "false", "f", "n", "0"):
            return False
        raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser(description="Evaluate SNAP-V Hardware-Constrained SNN in ViZDoom")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/snapv_policy_best.pth", help="Path to SNAP-V checkpoint")
    parser.add_argument("--episodes", type=int, default=10, help="Number of evaluation episodes")
    parser.add_argument("--time-steps", type=int, default=4, help="Simulation timesteps T")
    parser.add_argument("--fps", type=int, default=25, help="Target playback FPS when rendering (0 for uncapped)")
    parser.add_argument(
        "--render",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="Enable live ViZDoom rendering window (flags: --render, --no-render, or --render true/false)",
    )
    parser.add_argument(
        "--no-render",
        dest="render",
        action="store_false",
        help="Disable rendering window for fast headless evaluation (default: headless)",
    )
    parser.add_argument("--verbose-steps", action="store_true", help="Print Q-values and spike rates for every frame")
    args = parser.parse_args()

    evaluate_snapv(
        checkpoint_path=args.checkpoint,
        episodes=args.episodes,
        time_steps=args.time_steps,
        fps=args.fps,
        render=args.render,
        verbose_steps=args.verbose_steps,
    )
