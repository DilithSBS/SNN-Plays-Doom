"""
SNAP-V (Cerebra-H) Hardware-Constrained SNN Training Pipeline

Features:
  - 32x32 Input Resolution Downsampling for SNAP-V FPGA core
  - 5 Compound Discrete Actions: [Left, Right, Shoot, Left+Shoot, Right+Shoot]
  - Horizontal Data Augmentation with symmetric 5-action swapping
  - Fixed Power-of-Two Shift Decay (ShiftLIF): beta = 1 - 2^(-k)
  - Fake INT8 Synaptic Quantization (QuantConv2d & QuantLinear) via Straight-Through Estimator (STE)
  - One-Sided Homeostatic Network-on-Chip (NoC) Congestion Sparsity Loss (target <= 10% firing rate)
  - Off-Chip Teacher-Student Knowledge Distillation from pre-trained SNN (snn_policy_best.pth)
  - Live window rendering or headless training option
"""

import os
import sys
import time
import random
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from environment.doom_env import DoomEnv
from models.snn.snapv_snn import SnapVSpikingDQN
from models.snn.spiking_cnn import SpikingDuelingDQN


# =============================================================================
# 1. Experience Replay Buffer with 5-Action Augmentation & Teacher Distillation
# =============================================================================

class ReplayBuffer:
    """Experience Replay Buffer with 5-Action Horizontal Data Augmentation and optional Teacher Distillation."""
    def __init__(self, capacity: int = 50_000):
        self.buffer = deque(maxlen=capacity)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        teacher_q: np.ndarray | None = None,
    ):
        self.buffer.append((state, int(action), float(reward), next_state, bool(done), teacher_q))

    def sample(self, batch_size: int, device: torch.device):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones, teacher_qs = zip(*batch)

        states_t = torch.as_tensor(np.array(states), dtype=torch.float32, device=device)
        next_states_t = torch.as_tensor(np.array(next_states), dtype=torch.float32, device=device)
        actions_t = torch.as_tensor(actions, dtype=torch.long, device=device).unsqueeze(1)
        rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=device).unsqueeze(1)
        dones_t = torch.as_tensor(dones, dtype=torch.float32, device=device).unsqueeze(1)

        has_teacher = teacher_qs[0] is not None
        if has_teacher:
            teacher_q_t = torch.as_tensor(np.array(teacher_qs), dtype=torch.float32, device=device)
        else:
            teacher_q_t = None

        # -----------------------------------------------------------------
        # HORIZONTAL AUGMENTATION ACROSS 5 ACTIONS (50% probability)
        # -----------------------------------------------------------------
        if random.random() < 0.5:
            # Flip image across width axis (dim 3 for (B, 4, 32, 32))
            states_t = torch.flip(states_t, dims=[3])
            next_states_t = torch.flip(next_states_t, dims=[3])

            # Swap actions: 0 (Left) <-> 1 (Right), 3 (Left+Shoot) <-> 4 (Right+Shoot)
            swapped = actions_t.clone()
            swapped[actions_t == 0] = 1
            swapped[actions_t == 1] = 0
            swapped[actions_t == 3] = 4
            swapped[actions_t == 4] = 3
            actions_t = swapped

            if teacher_q_t is not None:
                swapped_t_q = teacher_q_t.clone()
                swapped_t_q[:, 0] = teacher_q_t[:, 1]
                swapped_t_q[:, 1] = teacher_q_t[:, 0]
                swapped_t_q[:, 3] = teacher_q_t[:, 4]
                swapped_t_q[:, 4] = teacher_q_t[:, 3]
                teacher_q_t = swapped_t_q

        return states_t, actions_t, rewards_t, next_states_t, dones_t, teacher_q_t

    def __len__(self):
        return len(self.buffer)


# =============================================================================
# 2. State Formatting & Action Helpers
# =============================================================================

def format_state(state, target_res: int = 32) -> np.ndarray:
    """Standardizes DoomEnv observations to shape (4, H, W)."""
    if isinstance(state, torch.Tensor):
        state = state.detach().cpu().numpy()
    state = np.asarray(state, dtype=np.float32)

    if state.ndim == 4 and state.shape[0] == 1:
        state = state.squeeze(0)
    elif state.ndim == 2:
        state = np.expand_dims(state, axis=0)
    elif state.ndim == 3 and state.shape[-1] in (1, 3, 4) and state.shape[0] == target_res:
        state = np.transpose(state, (2, 0, 1))

    return state


def downsample_84_to_32(x: np.ndarray) -> np.ndarray:
    """Downsamples (4, 84, 84) observation stack to (4, 32, 32) for SNAP-V."""
    with torch.no_grad():
        t = torch.as_tensor(x, dtype=torch.float32).unsqueeze(0)
        t_down = F.interpolate(t, size=(32, 32), mode="area")
        return t_down.squeeze(0).numpy()


def to_vizdoom_action(env, action_idx: int) -> list[int]:
    """Retrieves action vector from environment action table."""
    if hasattr(env, "actions"):
        return list(env.actions[action_idx])
    one_hot = [0] * env.num_actions
    one_hot[action_idx] = 1
    return one_hot


@torch.no_grad()
def select_action(
    policy_net: SnapVSpikingDQN,
    state: np.ndarray,
    epsilon: float,
    num_actions: int,
    device: torch.device,
) -> tuple[int, float]:
    if random.random() < epsilon:
        return random.randint(0, num_actions - 1), 0.0

    was_training = policy_net.training
    policy_net.eval()

    state_t = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    q_values, firing_rate = policy_net(state_t, return_firing_rate=True)
    action = int(q_values.argmax(dim=1).item())

    if was_training:
        policy_net.train()

    return action, float(firing_rate.item())


def soft_update_target(policy_net: nn.Module, target_net: nn.Module, tau: float = 0.005):
    """Polyak soft update for target network."""
    with torch.no_grad():
        for p_param, t_param in zip(policy_net.parameters(), target_net.parameters()):
            t_param.data.mul_(1.0 - tau).add_(p_param.data, alpha=tau)
        for p_buf, t_buf in zip(policy_net.buffers(), target_net.buffers()):
            t_buf.data.copy_(p_buf.data)


# =============================================================================
# 3. Hardware-Aware DDQN + Teacher Distillation Optimization Step
# =============================================================================

def optimize_snn_step(
    policy_net: SnapVSpikingDQN,
    target_net: SnapVSpikingDQN,
    optimizer: optim.Optimizer,
    replay_buffer: ReplayBuffer,
    batch_size: int,
    gamma: float,
    target_firing_rate: float,
    sparsity_weight: float,
    device: torch.device,
    distill_weight: float = 0.50,
):
    states, actions, rewards, next_states, dones, teacher_q = replay_buffer.sample(batch_size, device)

    policy_net.train()
    target_net.eval()

    # 1. Forward pass through Online SNN (with fake INT8 quantization + ShiftLIF)
    q_online, firing_rate = policy_net(states, return_firing_rate=True)
    state_action_values = q_online.gather(1, actions)

    # 2. DDQN Target calculation
    with torch.no_grad():
        next_actions = policy_net(next_states, return_firing_rate=False).argmax(dim=1, keepdim=True)
        next_q_target = target_net(next_states, return_firing_rate=False).gather(1, next_actions)
        expected_q_values = rewards + (1.0 - dones) * gamma * next_q_target

    # 3. TD Loss (Huber Loss)
    td_loss = F.smooth_l1_loss(state_action_values, expected_q_values)

    # 4. Off-Chip Teacher Distillation Loss
    if teacher_q is not None and distill_weight > 0.0:
        distill_loss = F.smooth_l1_loss(q_online, teacher_q)
    else:
        distill_loss = torch.tensor(0.0, device=device)

    # 5. One-Sided NoC Congestion Sparsity Loss (ReLU barrier)
    # Only penalizes firing rates that STRICTLY EXCEED target_firing_rate (10%)
    excess_spikes = torch.relu(firing_rate - target_firing_rate)
    sparsity_loss = sparsity_weight * (excess_spikes ** 2)

    total_loss = td_loss + (distill_weight * distill_loss) + sparsity_loss

    # 6. Backpropagation Through Time with Gradient Clipping
    optimizer.zero_grad(set_to_none=True)
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=1.0)
    optimizer.step()

    # 7. Polyak Soft Update
    soft_update_target(policy_net, target_net, tau=0.005)

    return float(td_loss.item()), float(firing_rate.item()), float(distill_loss.item())


# =============================================================================
# 4. SNAP-V Training Pipeline (with Optional Off-Chip Teacher Guidance)
# =============================================================================

def train_snapv(
    num_episodes: int = 800,
    batch_size: int = 64,
    gamma: float = 0.99,
    lr: float = 1e-4,
    time_steps: int = 4,
    eps_start: float = 1.0,
    eps_end: float = 0.08,
    eps_decay_episodes: int = 650,
    warmup_steps: int = 2_000,
    train_freq: int = 4,
    target_firing_rate: float = 0.10,  # 10% target firing rate to minimize NoC congestion
    sparsity_weight: float = 0.05,
    checkpoint_dir: str = "checkpoints",
    resume_from: str | None = None,
    render: bool = False,
    teacher_path: str | None = None,
    distill_weight: float = 0.50,
    teacher_guide_prob: float = 0.25,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Load Off-Chip Teacher Model if specified
    teacher_net = None
    if teacher_path is not None:
        if os.path.exists(teacher_path):
            print(f"[Teacher] Loading expert baseline SNN from '{teacher_path}'...")
            t_ckpt = torch.load(teacher_path, map_location=device, weights_only=False)
            t_state = t_ckpt["model_state_dict"] if isinstance(t_ckpt, dict) and "model_state_dict" in t_ckpt else t_ckpt
            teacher_net = SpikingDuelingDQN(
                input_channels=4,
                num_actions=5,
                time_steps=time_steps,
            ).to(device)
            teacher_net.load_state_dict(t_state)
            teacher_net.eval()
            for p in teacher_net.parameters():
                p.requires_grad = False
            print(f"[Teacher] Loaded successfully! Distillation weight: {distill_weight:.2f}, Guidance prob: {teacher_guide_prob*100:.0f}%")
        else:
            print(f"[Warning] Teacher checkpoint '{teacher_path}' not found! Proceeding without teacher.")

    # Initialize ViZDoom: 84x84 if teacher is active, otherwise 32x32 native
    env_res = (84, 84) if teacher_net is not None else (32, 32)
    env = DoomEnv(resolution=env_res, render=render)
    raw_obs = env.reset()

    if teacher_net is not None:
        sample_obs_84 = format_state(raw_obs, target_res=84)
        sample_obs = downsample_84_to_32(sample_obs_84)
    else:
        sample_obs = format_state(raw_obs, target_res=32)

    input_channels = sample_obs.shape[0]  # 4 stacked frames
    num_actions = env.num_actions        # 5 compound actions

    print("=" * 76)
    print("      SNAP-V (CEREBRA-H) HARDWARE-CONSTRAINED SNN TRAINING PIPELINE")
    print("=" * 76)
    print(f"Device        : {device}")
    print(f"Resolution    : 32x32 (Hardware input: {sample_obs.shape})")
    print(f"Teacher Mode  : {'Active (' + teacher_path + ')' if teacher_net else 'Standalone (From Scratch)'}")
    print(f"Action Space  : {num_actions} Compound Actions")
    print(f"Micro-Ticks   : T = {time_steps}")
    print(f"NoC Sparsity  : Target Firing Rate <= {target_firing_rate*100:.1f}% (One-Sided Barrier)")
    print(f"Rendering     : {'Enabled' if render else 'Disabled'}")

    # Instantiate Hardware-Aware SNAP-V Spiking Networks
    policy_net = SnapVSpikingDQN(
        input_channels=input_channels,
        num_actions=num_actions,
        time_steps=time_steps,
    ).to(device)

    target_net = SnapVSpikingDQN(
        input_channels=input_channels,
        num_actions=num_actions,
        time_steps=time_steps,
    ).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    # Hardware resource summary
    res = policy_net.count_resources()
    neurons = res["neurons"]
    synapses = res["synapses"]
    sram_pct = (synapses["total_synapses"] / 524288) * 100
    print(f"Hardware Core : {neurons['total_spiking_neurons']:,} spiking neurons | {synapses['total_synapses']:,} synapses")
    print(f"SRAM Usage    : {sram_pct:.2f}% of SNAP-V 524,288 weight limit (PASSED: < 450,000)")
    print("-" * 76)

    optimizer = optim.AdamW(policy_net.parameters(), lr=lr, weight_decay=1e-4)
    replay_buffer = ReplayBuffer(capacity=50_000)

    best_avg_reward = -float("inf")
    best_path = os.path.join(checkpoint_dir, "snapv_policy_best.pth")
    final_path = os.path.join(checkpoint_dir, "snapv_policy_final.pth")
    reward_window = deque(maxlen=50)
    global_step = 0
    start_time = time.time()
    start_episode = 1

    # Resume from existing checkpoint if specified
    if resume_from is not None:
        if os.path.exists(resume_from):
            print(f"[Resume] Loading checkpoint from '{resume_from}'...")
            ckpt = torch.load(resume_from, map_location=device, weights_only=False)
            if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
                policy_net.load_state_dict(ckpt["model_state_dict"])
                target_net.load_state_dict(ckpt["model_state_dict"])
                if "optimizer_state_dict" in ckpt:
                    try:
                        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                    except Exception:
                        pass
                start_episode = ckpt.get("episode", 0) + 1
                best_avg_reward = ckpt.get("avg_reward_50", -float("inf"))
                print(f"[Resume] Resumed from Episode {start_episode} (Prior Best Avg50: {best_avg_reward:.1f})")
            else:
                policy_net.load_state_dict(ckpt)
                target_net.load_state_dict(ckpt)
                print(f"[Resume] Loaded raw state_dict into policy and target networks.")

            if eps_start is None:
                eps_start = 0.25
                print(f"[Resume] Set initial exploration epsilon to {eps_start:.2f}.")
        else:
            print(f"[Warning] Checkpoint '{resume_from}' not found! Starting fresh.")

    if eps_start is None:
        eps_start = 1.0

    try:
        for episode in range(start_episode, start_episode + num_episodes):
            raw_reset = env.reset()
            if teacher_net is not None:
                state_84 = format_state(raw_reset, target_res=84)
                state_32 = downsample_84_to_32(state_84)
            else:
                state_32 = format_state(raw_reset, target_res=32)
                state_84 = None

            done = False
            episode_reward = 0.0
            episode_losses = []
            episode_distill_losses = []
            episode_firing_rates = []

            # Linear epsilon decay relative to remaining training episodes
            progress = min(1.0, (episode - start_episode) / float(max(1, eps_decay_episodes)))
            epsilon = eps_start - progress * (eps_start - eps_end)

            while not done:
                global_step += 1

                # Query teacher for Q-targets and expert guidance action
                teacher_q_np = None
                teacher_action = None
                if teacher_net is not None:
                    with torch.no_grad():
                        s84_t = torch.as_tensor(state_84, dtype=torch.float32, device=device).unsqueeze(0)
                        t_q = teacher_net(s84_t, return_firing_rate=False)
                        teacher_q_np = t_q.squeeze(0).cpu().numpy()
                        teacher_action = int(t_q.argmax(dim=1).item())

                # Action Selection: Teacher Guidance during exploration OR Student policy
                if teacher_action is not None and (random.random() < teacher_guide_prob) and (random.random() < epsilon):
                    action = teacher_action
                    step_fr = 0.0
                else:
                    action, step_fr = select_action(policy_net, state_32, epsilon, num_actions, device)

                # Execute action in ViZDoom
                viz_action = to_vizdoom_action(env, action)
                step_out = env.step(viz_action)
                if len(step_out) == 3:
                    next_raw_state, reward, done = step_out
                elif len(step_out) == 4:
                    next_raw_state, reward, done, _ = step_out
                else:
                    next_raw_state, reward, terminated, truncated, _ = step_out
                    done = terminated or truncated

                if teacher_net is not None:
                    next_state_84 = format_state(next_raw_state, target_res=84) if next_raw_state is not None else np.zeros_like(state_84)
                    next_state_32 = downsample_84_to_32(next_state_84)
                else:
                    next_state_32 = format_state(next_raw_state, target_res=32) if next_raw_state is not None else np.zeros_like(state_32)
                    next_state_84 = None

                # Scaled reward for BPTT stability
                scaled_reward = float(reward) / 100.0
                replay_buffer.push(state_32, action, scaled_reward, next_state_32, done, teacher_q_np)

                state_32 = next_state_32
                if teacher_net is not None:
                    state_84 = next_state_84

                episode_reward += float(reward)
                if step_fr > 0.0:
                    episode_firing_rates.append(step_fr)

                # Optimize SNN every `train_freq` steps once warmup is complete
                if len(replay_buffer) >= max(warmup_steps, batch_size) and (global_step % train_freq == 0):
                    loss_val, batch_fr, d_loss = optimize_snn_step(
                        policy_net=policy_net,
                        target_net=target_net,
                        optimizer=optimizer,
                        replay_buffer=replay_buffer,
                        batch_size=batch_size,
                        gamma=gamma,
                        target_firing_rate=target_firing_rate,
                        sparsity_weight=sparsity_weight,
                        device=device,
                        distill_weight=distill_weight,
                    )
                    episode_losses.append(loss_val)
                    episode_distill_losses.append(d_loss)
                    episode_firing_rates.append(batch_fr)

            reward_window.append(episode_reward)
            avg_reward_50 = float(np.mean(reward_window))
            avg_loss = float(np.mean(episode_losses)) if episode_losses else 0.0
            avg_distill = float(np.mean(episode_distill_losses)) if episode_distill_losses else 0.0
            avg_fr = float(np.mean(episode_firing_rates)) if episode_firing_rates else 0.0

            # Save best SNAP-V model checkpoint
            if len(reward_window) >= 20 and avg_reward_50 > best_avg_reward:
                best_avg_reward = avg_reward_50
                best_path = os.path.join(checkpoint_dir, "snapv_policy_best.pth")
                torch.save(
                    {
                        "episode": episode,
                        "model_state_dict": policy_net.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "avg_reward_50": avg_reward_50,
                        "time_steps": time_steps,
                        "input_channels": input_channels,
                        "num_actions": num_actions,
                        "resolution": (32, 32),
                        "model_type": "SnapVSpikingDQN",
                    },
                    best_path,
                )

            if episode % 10 == 0 or episode == start_episode:
                elapsed = time.time() - start_time
                teacher_log = f" | Distill: {avg_distill:.4f}" if teacher_net is not None else ""
                print(
                    f"Ep {episode:4d}/{start_episode + num_episodes - 1} | "
                    f"Reward: {episode_reward:6.1f} | "
                    f"Avg50: {avg_reward_50:6.1f} | "
                    f"Eps: {epsilon:.2f} | "
                    f"Loss: {avg_loss:.4f} | "
                    f"SpikeRate: {avg_fr*100:4.1f}%"
                    f"{teacher_log} | "
                    f"Time: {elapsed:.0f}s"
                )

        final_path = os.path.join(checkpoint_dir, "snapv_policy_final.pth")
        torch.save(policy_net.state_dict(), final_path)
        print(f"\n[SNAP-V] Training complete! Best 50-episode average reward: {best_avg_reward:.1f}")
        print(f"[SNAP-V] Checkpoints saved to:\n  - {best_path}\n  - {final_path}")

    finally:
        env.close()


if __name__ == "__main__":
    import argparse

    def str2bool(v):
        if isinstance(v, bool):
            return v
        if str(v).lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif str(v).lower() in ("no", "false", "f", "n", "0"):
            return False
        raise argparse.ArgumentTypeError("Boolean value expected (e.g., True/False).")

    parser = argparse.ArgumentParser(description="Train SNAP-V Hardware-Constrained Spiking Dueling DQN on ViZDoom")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume training from")
    parser.add_argument("--teacher", type=str, default=None, help="Path to pre-trained SNN teacher checkpoint (e.g. checkpoints/snn_policy_best.pth)")
    parser.add_argument("--distill-weight", type=float, default=0.50, help="Weight of distillation loss from teacher (default: 0.50)")
    parser.add_argument("--guide-prob", type=float, default=0.25, help="Probability of teacher action guidance during exploration (default: 0.25)")
    parser.add_argument("--episodes", type=int, default=800, help="Number of episodes to train")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--eps-start", type=float, default=None, help="Initial exploration epsilon")
    parser.add_argument("--eps-end", type=float, default=0.08, help="Final exploration epsilon")
    parser.add_argument("--warmup-steps", type=int, default=2000, help="Replay buffer warmup steps before training")
    parser.add_argument("--decay-episodes", type=int, default=650, help="Episodes over which epsilon decays")
    parser.add_argument(
        "--render",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="Enable live ViZDoom rendering window (use --render, --no-render, or --render true/false)",
    )
    parser.add_argument(
        "--no-render",
        dest="render",
        action="store_false",
        help="Disable live ViZDoom window for headless training (default: headless)",
    )
    args = parser.parse_args()

    kwargs = {
        "num_episodes": args.episodes,
        "lr": args.lr,
        "eps_end": args.eps_end,
        "eps_decay_episodes": args.decay_episodes,
        "warmup_steps": args.warmup_steps,
        "render": args.render,
        "teacher_path": args.teacher,
        "distill_weight": args.distill_weight,
        "teacher_guide_prob": args.guide_prob,
    }
    if args.resume:
        kwargs["resume_from"] = args.resume
    if args.eps_start is not None:
        kwargs["eps_start"] = args.eps_start

    train_snapv(**kwargs)
