import os
import sys
import time
import argparse
import numpy as np
import torch

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from environment.doom_env import DoomEnv
from models.snn.spiking_cnn import SpikingDuelingDQN, ParametricLIF


class SpikeTelemetryMonitor:
    """
    Non-intrusive PyTorch forward-hook manager that records exact layer-by-layer
    spike counts, firing rates, and temporal spike trains across T timesteps.
    """
    def __init__(self, model: SpikingDuelingDQN):
        self.model = model
        self.hooks = []
        self.layer_spikes = {}
        self.layer_sizes = {}
        self._register_hooks()

    def _register_hooks(self):
        for name, module in self.model.named_modules():
            if isinstance(module, ParametricLIF):
                self.layer_spikes[name] = []
                hook = module.register_forward_hook(self._make_hook(name))
                self.hooks.append(hook)

    def _make_hook(self, layer_name: str):
        def hook_fn(module, inputs, outputs):
            # ParametricLIF returns (spike, membrane)
            spike = outputs[0].detach()
            self.layer_spikes[layer_name].append(spike.sum().item())
            self.layer_sizes[layer_name] = spike.numel()
        return hook_fn

    def reset_step(self):
        for k in self.layer_spikes:
            self.layer_spikes[k].clear()

    def get_step_summary(self) -> dict:
        """
        Returns per-layer firing rate (0.0 to 1.0) and total spike count
        for the most recent forward pass (aggregated across T timesteps).
        """
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
                "temporal_profile": spk_list,  # Spikes at each t = 0..T-1
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
    """Standardizes DoomEnv observation to (C, 84, 84)."""
    if isinstance(state, torch.Tensor):
        state = state.detach().cpu().numpy()
    state = np.asarray(state)

    if state.ndim == 4 and state.shape[0] == 1:
        state = state.squeeze(0)
    elif state.ndim == 2:
        state = np.expand_dims(state, axis=0)
    elif state.ndim == 3 and state.shape[-1] in (1, 3, 4) and state.shape[0] == 84:
        state = np.transpose(state, (2, 0, 1))

    return state


def ascii_bar(rate: float, width: int = 15) -> str:
    """Renders a compact ASCII progress bar for firing rate percentage."""
    # Scale bar up to 40% firing rate as full width for visual clarity
    normalized = min(1.0, max(0.0, rate / 0.40))
    filled = int(round(normalized * width))
    return "█" * filled + "░" * (width - filled)


def load_snn_checkpoint(checkpoint_path: str, input_channels: int, num_actions: int, default_t: int, device: torch.device):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found at '{checkpoint_path}'. "
            f"Run `python training/train_snn.py` first!"
        )

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Support both full dictionary checkpoints and raw state_dict saves
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        time_steps = ckpt.get("time_steps", default_t)
        state_dict = ckpt["model_state_dict"]
        meta_info = f"(Trained Ep: {ckpt.get('episode', '?')}, Best Avg50: {ckpt.get('avg_reward_50', 0.0):.1f})"
    else:
        time_steps = default_t
        state_dict = ckpt
        meta_info = "(Raw state_dict)"

    model = SpikingDuelingDQN(
        input_channels=input_channels,
        num_actions=num_actions,
        time_steps=time_steps,
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    print(f"[Loaded SNN] {checkpoint_path} {meta_info} | T={time_steps}")
    return model


@torch.no_grad()
def evaluate_snn(
    checkpoint_path: str = "checkpoints/snn_policy_best.pth",
    episodes: int = 10,
    time_steps: int = 4,
    fps: int = 25,
    render: bool = True,
    verbose_steps: bool = False,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize ViZDoom with live window rendering
    env = DoomEnv(render=render)
    sample_obs = format_state(env.reset())
    input_channels = sample_obs.shape[0]
    num_actions = getattr(env, "num_actions", 3)
    action_names = {0: "LEFT ", 1: "RIGHT", 2: "SHOOT"}

    model = load_snn_checkpoint(
        checkpoint_path=checkpoint_path,
        input_channels=input_channels,
        num_actions=num_actions,
        default_t=time_steps,
        device=device,
    )

    # Display learned PLIF membrane time constants (beta)
    betas = {
        "lif1": model.lif1.beta.mean().item(),
        "lif2": model.lif2.beta.mean().item(),
        "lif3": model.lif3.beta.mean().item(),
        "lif_fc": model.lif_fc.beta.mean().item(),
    }
    print(
        f"[Learned PLIF Betas] "
        f"lif1: {betas['lif1']:.3f} | lif2: {betas['lif2']:.3f} | "
        f"lif3: {betas['lif3']:.3f} | lif_fc: {betas['lif_fc']:.3f}\n"
        + "-" * 78
    )

    monitor = SpikeTelemetryMonitor(model)
    frame_delay = (1.0 / fps) if (render and fps > 0) else 0.0

    episode_rewards = []
    episode_Steps = []
    global_layer_rates = {name: [] for name in ["lif1", "lif2", "lif3", "lif_fc", "total"]}

    try:
        for ep in range(1, episodes + 1):
            state = format_state(env.reset())
            done = False
            ep_reward = 0.0
            steps = 0

            ep_layer_accum = {name: [] for name in ["lif1", "lif2", "lif3", "lif_fc", "total"]}

            while not done:
                steps += 1
                monitor.reset_step()

                state_t = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                q_values = model(state_t, return_firing_rate=False)
                action = int(q_values.argmax(dim=1).item())

                step_stats = monitor.get_step_summary()
                for k in ep_layer_accum:
                    if k in step_stats:
                        ep_layer_accum[k].append(step_stats[k]["rate"])

                if verbose_steps:
                    act_str = action_names.get(action, f"ACT_{action}")
                    q_str = ", ".join([f"{v:+.2f}" for v in q_values.squeeze(0).tolist()])
                    print(
                        f"  Step {steps:3d} | Action: {act_str} | Q: [{q_str}] | "
                        f"L1:{step_stats['lif1']['rate']*100:4.1f}% "
                        f"L2:{step_stats['lif2']['rate']*100:4.1f}% "
                        f"L3:{step_stats['lif3']['rate']*100:4.1f}% "
                        f"FC:{step_stats['lif_fc']['rate']*100:4.1f}%"
                    )

                step_out = env.step(action)
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
            episode_Steps.append(steps)

            # Aggregate episode spike statistics
            ep_means = {k: float(np.mean(v)) if v else 0.0 for k, v in ep_layer_accum.items()}
            for k, val in ep_means.items():
                global_layer_rates[k].append(val)

            print(
                f"Episode {ep:2d}/{episodes} | "
                f"Reward: {ep_reward:6.1f} | "
                f"Steps: {steps:3d} | "
                f"Spikes -> "
                f"L1: {ep_means['lif1']*100:4.1f}% | "
                f"L2: {ep_means['lif2']*100:4.1f}% | "
                f"L3: {ep_means['lif3']*100:4.1f}% | "
                f"FC: {ep_means['lif_fc']*100:4.1f}% | "
                f"All: {ep_means['total']*100:4.1f}%"
            )

        # Print Final Neuromorphic Benchmark Report
        print("\n" + "=" * 78)
        print("                SNN VIZDOOM EVALUATION & SPIKE TELEMETRY REPORT")
        print("=" * 78)
        print(f"  Episodes Evaluated : {episodes}")
        print(f"  Mean Game Reward   : {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
        print(f"  Max / Min Reward   : {np.max(episode_rewards):.1f} / {np.min(episode_rewards):.1f}")
        print(f"  Mean Survival Steps: {np.mean(episode_Steps):.1f} steps")
        print("-" * 78)
        print("  Layer-by-Layer Spiking Activity (Mean across all episodes & timesteps):")

        layer_labels = [
            ("lif1", "Conv1 + PLIF (32x20x20 = 12,800 neurons)"),
            ("lif2", "Conv2 + PLIF (64x9x9   =  5,184 neurons)"),
            ("lif3", "Conv3 + PLIF (64x7x7   =  3,136 neurons)"),
            ("lif_fc", "FC1   + PLIF (512      =    512 neurons)"),
            ("total", "OVERALL NETWORK SPARSITY (21,632 neurons)"),
        ]

        for key, desc in layer_labels:
            mean_rate = float(np.mean(global_layer_rates[key]))
            bar = ascii_bar(mean_rate)
            if key == "total":
                print("-" * 78)
            print(f"  {desc:<43} : {bar} {mean_rate*100:5.2f}%")

        overall_sparsity = 1.0 - float(np.mean(global_layer_rates["total"]))
        print(f"\n  Neuromorphic Event Sparsity: {overall_sparsity*100:.1f}% of neuron-timesteps are silent (0 spikes).")
        print("=" * 78)

    finally:
        monitor.close()
        if hasattr(env, "close"):
            env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained Spiking Dueling DQN in ViZDoom")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/snn_policy_best.pth", help="Path to SNN checkpoint")
    parser.add_argument("--episodes", type=int, default=10, help="Number of evaluation episodes")
    parser.add_argument("--time-steps", type=int, default=4, help="Fallback simulation timesteps T")
    parser.add_argument("--fps", type=int, default=25, help="Playback FPS when rendering (0 for uncapped)")
    parser.add_argument("--no-render", action="store_true", help="Disable live ViZDoom window for fast benchmarking")
    parser.add_argument("--verbose-steps", action="store_true", help="Print Q-values and spike rates at every single frame")
    args = parser.parse_args()

    evaluate_snn(
        checkpoint_path=args.checkpoint,
        episodes=args.episodes,
        time_steps=args.time_steps,
        fps=args.fps,
        render=not args.no_render,
        verbose_steps=args.verbose_steps,
    )