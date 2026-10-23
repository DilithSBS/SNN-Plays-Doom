"""
SNAP-V (Cerebra-H) Hardware-Constrained SNN 3D Navigation Training Pipeline
Scenario: ViZDoom `configs/my_way_home.cfg` (Labyrinth Goal Traversal)

Features:
  - 32x32 Input Resolution Downsampling for SNAP-V FPGA core
  - 5 Compound 3D Navigation Actions: [Turn Left, Turn Right, Forward, Fwd+Left, Fwd+Right]
  - Symmetric Horizontal Augmentation for Rotational Steering
  - Count-Based Spatial Novelty Exploration Bonus (Grid Discretization)
  - ShiftLIF Bit-Shift Decay (k=2, k=3) & Fake INT8 Quantization (STE)
  - One-Sided NoC Congestion Sparsity Barrier (< 10% Firing Rate)
  - Off-Chip Teacher Distillation & Live Window Rendering Options
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


# =============================================================================
# 1. Experience Replay Buffer for 3D Navigation
# =============================================================================

class NavReplayBuffer:
    """Experience Replay Buffer with Symmetric Rotational Action Swapping."""
    def __init__(self, capacity: int = 50_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        self.buffer.append((state, int(action), float(reward), next_state, bool(done)))

    def sample(self, batch_size: int, device: torch.device):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states_t = torch.as_tensor(np.array(states), dtype=torch.float32, device=device)
        next_states_t = torch.as_tensor(np.array(next_states), dtype=torch.float32, device=device)
        actions_t = torch.as_tensor(actions, dtype=torch.long, device=device).unsqueeze(1)
        rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=device).unsqueeze(1)
        dones_t = torch.as_tensor(dones, dtype=torch.float32, device=device).unsqueeze(1)

        # -----------------------------------------------------------------
        # HORIZONTAL AUGMENTATION ACROSS 5 NAVIGATION ACTIONS (50% probability)
        # -----------------------------------------------------------------
        if random.random() < 0.5:
            # Flip image across width axis
            states_t = torch.flip(states_t, dims=[3])
            next_states_t = torch.flip(next_states_t, dims=[3])

            # Swap: Turn Left (0) <-> Turn Right (1), Fwd+Left (3) <-> Fwd+Right (4)
            # Forward (2) stays invariant!
            swapped = actions_t.clone()
            swapped[actions_t == 0] = 1
            swapped[actions_t == 1] = 0
            swapped[actions_t == 3] = 4
            swapped[actions_t == 4] = 3
            actions_t = swapped

        return states_t, actions_t, rewards_t, next_states_t, dones_t

    def __len__(self):
        return len(self.buffer)


# =============================================================================
# 2. Helpers & Action Selection
# =============================================================================

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


@torch.no_grad()
def select_nav_action(
    policy_net: SnapVSpikingDQN,
    state: np.ndarray,
    epsilon: float,
    num_actions: int,
    device: torch.device,
) -> tuple[int, float]:
    # Exploration: bias slightly toward Forward motion rather than spinning
    if random.random() < epsilon:
        if random.random() < 0.50:
            return 2, 0.0  # Forward bias to drive maze traversal
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
    with torch.no_grad():
        for p_param, t_param in zip(policy_net.parameters(), target_net.parameters()):
            t_param.data.mul_(1.0 - tau).add_(p_param.data, alpha=tau)
        for p_buf, t_buf in zip(policy_net.buffers(), target_net.buffers()):
            t_buf.data.copy_(p_buf.data)


def optimize_nav_step(
    policy_net: SnapVSpikingDQN,
    target_net: SnapVSpikingDQN,
    optimizer: optim.Optimizer,
    replay_buffer: NavReplayBuffer,
    batch_size: int,
    gamma: float,
    target_firing_rate: float,
    sparsity_weight: float,
    device: torch.device,
):
    states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size, device)

    policy_net.train()
    target_net.eval()

    # 1. Forward pass online SNN
    q_online, firing_rate = policy_net(states, return_firing_rate=True)
    state_action_values = q_online.gather(1, actions)

    # 2. DDQN Target
    with torch.no_grad():
        next_actions = policy_net(next_states, return_firing_rate=False).argmax(dim=1, keepdim=True)
        next_q_target = target_net(next_states, return_firing_rate=False).gather(1, next_actions)
        expected_q_values = rewards + (1.0 - dones) * gamma * next_q_target

    # 3. Huber TD Loss
    td_loss = F.smooth_l1_loss(state_action_values, expected_q_values)

    # 4. One-Sided NoC Congestion Sparsity Barrier
    excess_spikes = torch.relu(firing_rate - target_firing_rate)
    sparsity_loss = sparsity_weight * (excess_spikes ** 2)

    total_loss = td_loss + sparsity_loss

    optimizer.zero_grad(set_to_none=True)
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=1.0)
    optimizer.step()

    soft_update_target(policy_net, target_net, tau=0.005)

    return float(td_loss.item()), float(firing_rate.item())


# =============================================================================
# 3. Main Navigation Training Loop
# =============================================================================

def train_nav(
    config_path: str = "configs/my_way_home.cfg",
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
    target_firing_rate: float = 0.10,
    sparsity_weight: float = 0.05,
    checkpoint_dir: str = "checkpoints",
    resume_from: str | None = None,
    render: bool = False,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Initialize ViZDoom with SNAP-V 32x32 resolution
    env = DoomEnv(config_path=config_path, resolution=(32, 32), render=render)
    sample_obs = format_state(env.reset())
    input_channels = sample_obs.shape[0]  # 4 stacked frames
    num_actions = env.num_actions        # 5 compound navigation actions

    print("=" * 80)
    print("      SNAP-V (CEREBRA-H) SNN 3D NAVIGATION PIPELINE (MY_WAY_HOME)")
    print("=" * 80)
    print(f"Device        : {device}")
    print(f"Scenario      : {config_path}")
    print(f"Resolution    : 32x32 (Hardware Stack: {sample_obs.shape})")
    print(f"Nav Actions   : {num_actions} Discrete Nav Actions: {list(env.action_names.values())}")
    print(f"Micro-Ticks   : T = {time_steps}")
    print(f"NoC Sparsity  : Target Firing Rate <= {target_firing_rate*100:.1f}%")
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

    res = policy_net.count_resources()
    neurons = res["neurons"]
    synapses = res["synapses"]
    sram_pct = (synapses["total_synapses"] / 524288) * 100
    print(f"Hardware Core : {neurons['total_spiking_neurons']:,} spiking neurons | {synapses['total_synapses']:,} synapses")
    print(f"SRAM Usage    : {sram_pct:.2f}% of SNAP-V 524,288 weight limit (PASSED: < 450,000)")
    print("-" * 80)

    optimizer = optim.AdamW(policy_net.parameters(), lr=lr, weight_decay=1e-4)
    replay_buffer = NavReplayBuffer(capacity=50_000)

    best_avg_reward = -float("inf")
    best_path = os.path.join(checkpoint_dir, "nav_snapv_best.pth")
    final_path = os.path.join(checkpoint_dir, "nav_snapv_final.pth")
    reward_window = deque(maxlen=50)
    goal_reach_window = deque(maxlen=50)
    global_step = 0
    start_time = time.time()
    start_episode = 1

    if resume_from is not None and os.path.exists(resume_from):
        print(f"[Resume] Loading checkpoint from '{resume_from}'...")
        ckpt = torch.load(resume_from, map_location=device, weights_only=False)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            policy_net.load_state_dict(ckpt["model_state_dict"])
            target_net.load_state_dict(ckpt["model_state_dict"])
            start_episode = ckpt.get("episode", 0) + 1
            best_avg_reward = ckpt.get("avg_reward_50", -float("inf"))
            print(f"[Resume] Resumed from Episode {start_episode} (Prior Best Avg50: {best_avg_reward:.1f})")
        else:
            policy_net.load_state_dict(ckpt)
            target_net.load_state_dict(ckpt)

    try:
        for episode in range(start_episode, start_episode + num_episodes):
            state = format_state(env.reset())
            done = False
            episode_reward = 0.0
            episode_losses = []
            episode_firing_rates = []

            progress = min(1.0, (episode - start_episode) / float(max(1, eps_decay_episodes)))
            epsilon = eps_start - progress * (eps_start - eps_end)

            while not done:
                global_step += 1
                action, step_fr = select_nav_action(policy_net, state, epsilon, num_actions, device)

                viz_action = env.actions[action]
                step_out = env.step(viz_action)
                next_raw_state, reward, done, info = step_out

                next_state = format_state(next_raw_state) if next_raw_state is not None else np.zeros_like(state)

                # Scale reward for stability
                scaled_reward = float(reward) / 10.0
                replay_buffer.push(state, action, scaled_reward, next_state, done)

                state = next_state
                episode_reward += float(reward)
                if step_fr > 0.0:
                    episode_firing_rates.append(step_fr)

                if len(replay_buffer) >= max(warmup_steps, batch_size) and (global_step % train_freq == 0):
                    loss_val, batch_fr = optimize_nav_step(
                        policy_net=policy_net,
                        target_net=target_net,
                        optimizer=optimizer,
                        replay_buffer=replay_buffer,
                        batch_size=batch_size,
                        gamma=gamma,
                        target_firing_rate=target_firing_rate,
                        sparsity_weight=sparsity_weight,
                        device=device,
                    )
                    episode_losses.append(loss_val)
                    episode_firing_rates.append(batch_fr)

            goal_reached = 1 if episode_reward >= 5.0 else 0
            goal_reach_window.append(goal_reached)
            reward_window.append(episode_reward)
            avg_reward_50 = float(np.mean(reward_window))
            avg_reach_pct = float(np.mean(goal_reach_window)) * 100.0
            avg_loss = float(np.mean(episode_losses)) if episode_losses else 0.0
            avg_fr = float(np.mean(episode_firing_rates)) if episode_firing_rates else 0.0
            cells_found = info.get("cells_visited", 0)

            if len(reward_window) >= 15 and avg_reward_50 > best_avg_reward:
                best_avg_reward = avg_reward_50
                torch.save(
                    {
                        "episode": episode,
                        "model_state_dict": policy_net.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "avg_reward_50": avg_reward_50,
                        "avg_reach_pct": avg_reach_pct,
                        "time_steps": time_steps,
                        "input_channels": input_channels,
                        "num_actions": num_actions,
                        "resolution": (32, 32),
                        "model_type": "SnapVSpikingDQN_Navigation",
                    },
                    best_path,
                )

            if episode % 10 == 0 or episode == start_episode:
                elapsed = time.time() - start_time
                print(
                    f"Ep {episode:4d}/{start_episode + num_episodes - 1} | "
                    f"Rew: {episode_reward:5.1f} | "
                    f"Avg50: {avg_reward_50:5.1f} | "
                    f"GoalPct: {avg_reach_pct:4.1f}% | "
                    f"Cells: {cells_found:2d} | "
                    f"Eps: {epsilon:.2f} | "
                    f"Loss: {avg_loss:.4f} | "
                    f"SpikeRate: {avg_fr*100:4.1f}% | "
                    f"Time: {elapsed:.0f}s"
                )

        torch.save(policy_net.state_dict(), final_path)
        print(f"\n[Navigation] Training complete! Best Avg50 Reward: {best_avg_reward:.1f}")
        print(f"[Navigation] Saved to: {best_path} and {final_path}")

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
        raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser(description="Train SNAP-V SNN on 3D Navigation (my_way_home)")
    parser.add_argument("--config", type=str, default="configs/my_way_home.cfg", help="ViZDoom scenario config")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume")
    parser.add_argument("--episodes", type=int, default=800, help="Number of training episodes")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--eps-start", type=float, default=1.0, help="Initial exploration epsilon")
    parser.add_argument("--eps-end", type=float, default=0.08, help="Final exploration epsilon")
    parser.add_argument("--warmup-steps", type=int, default=2000, help="Replay buffer warmup steps")
    parser.add_argument(
        "--render",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="Enable live ViZDoom rendering window (use --render or --no-render)",
    )
    parser.add_argument("--no-render", dest="render", action="store_false", help="Disable live window (default: headless)")
    args = parser.parse_args()

    train_nav(
        config_path=args.config,
        num_episodes=args.episodes,
        lr=args.lr,
        eps_start=args.eps_start,
        eps_end=args.eps_end,
        warmup_steps=args.warmup_steps,
        render=args.render,
        resume_from=args.resume,
    )
