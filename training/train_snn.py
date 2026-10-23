import os
import sys
import time
import random
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from environment.doom_env import DoomEnv
from models.snn.spiking_cnn import SpikingDuelingDQN


class ReplayBuffer:
    """Experience Replay Buffer with 5-Action Horizontal Data Augmentation."""
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
        # HORIZONTAL AUGMENTATION ACROSS 5 ACTIONS (50% probability)
        # -----------------------------------------------------------------
        if random.random() < 0.5:
            # Flip image across width axis (dim 3 for (B, 4, 84, 84))
            states_t = torch.flip(states_t, dims=[3])
            next_states_t = torch.flip(next_states_t, dims=[3])

            # Swap: 0 (Left) <-> 1 (Right), 3 (Left+Shoot) <-> 4 (Right+Shoot)
            swapped = actions_t.clone()
            swapped[actions_t == 0] = 1
            swapped[actions_t == 1] = 0
            swapped[actions_t == 3] = 4
            swapped[actions_t == 4] = 3
            actions_t = swapped

        return states_t, actions_t, rewards_t, next_states_t, dones_t

    def __len__(self):
        return len(self.buffer)


def format_state(state) -> np.ndarray:
    """Standardizes DoomEnv observations to shape (4, 84, 84)."""
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


def to_vizdoom_action(env, action_idx: int) -> list[int]:
    """Retrieves action vector from environment action table."""
    if hasattr(env, "actions"):
        return list(env.actions[action_idx])
    one_hot = [0] * env.num_actions
    one_hot[action_idx] = 1
    return one_hot


@torch.no_grad()
def select_action(
    policy_net: SpikingDuelingDQN,
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
    with torch.no_grad():
        for p_param, t_param in zip(policy_net.parameters(), target_net.parameters()):
            t_param.data.mul_(1.0 - tau).add_(p_param.data, alpha=tau)
        for p_buf, t_buf in zip(policy_net.buffers(), target_net.buffers()):
            t_buf.data.copy_(p_buf.data)


def optimize_snn_step(
    policy_net: SpikingDuelingDQN,
    target_net: SpikingDuelingDQN,
    optimizer: optim.Optimizer,
    replay_buffer: ReplayBuffer,
    batch_size: int,
    gamma: float,
    target_firing_rate: float,
    sparsity_weight: float,
    device: torch.device,
):
    states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size, device)

    policy_net.train()
    target_net.eval()

    # 1. Forward pass through Online SNN
    q_online, firing_rate = policy_net(states, return_firing_rate=True)
    state_action_values = q_online.gather(1, actions)

    # 2. DDQN Target with target network
    with torch.no_grad():
        next_actions = policy_net(next_states, return_firing_rate=False).argmax(dim=1, keepdim=True)
        next_q_target = target_net(next_states, return_firing_rate=False).gather(1, next_actions)
        expected_q_values = rewards + (1.0 - dones) * gamma * next_q_target

    # 3. Loss = Huber Loss + Firing Rate Regularization
    td_loss = nn.functional.smooth_l1_loss(state_action_values, expected_q_values)
    sparsity_loss = sparsity_weight * ((firing_rate - target_firing_rate) ** 2)
    total_loss = td_loss + sparsity_loss

    # 4. Backpropagation Through Time
    optimizer.zero_grad(set_to_none=True)
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=1.0)
    optimizer.step()

    # 5. Polyak Soft Update
    soft_update_target(policy_net, target_net, tau=0.005)

    return float(td_loss.item()), float(firing_rate.item())


def train_snn(
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
    target_firing_rate: float = 0.12,
    sparsity_weight: float = 0.05,
    checkpoint_dir: str = "checkpoints",
    resume_from: str | None = None,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(checkpoint_dir, exist_ok=True)

    env = DoomEnv(render=False)
    sample_obs = format_state(env.reset())
    input_channels = sample_obs.shape[0]  # Automatically detects 4 channels
    num_actions = env.num_actions        # Automatically detects 5 actions

    print(f"[SNN-Doom] Device: {device} | Channels: {input_channels} | Actions: {num_actions} | T={time_steps}")

    policy_net = SpikingDuelingDQN(
        input_channels=input_channels,
        num_actions=num_actions,
        time_steps=time_steps,
    ).to(device)

    target_net = SpikingDuelingDQN(
        input_channels=input_channels,
        num_actions=num_actions,
        time_steps=time_steps,
    ).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.AdamW(policy_net.parameters(), lr=lr, weight_decay=1e-4)
    replay_buffer = ReplayBuffer(capacity=50_000)

    best_avg_reward = -float("inf")
    reward_window = deque(maxlen=50)
    global_step = 0
    start_time = time.time()
    start_episode = 1

    # Load from previous checkpoint if specified
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
                print(f"[Resume] Successfully loaded! Resuming from Episode {start_episode} (Prior Best Avg50: {best_avg_reward:.1f})")
            else:
                policy_net.load_state_dict(ckpt)
                target_net.load_state_dict(ckpt)
                print(f"[Resume] Successfully loaded raw state_dict into policy and target networks.")

            # Default to controlled exploration (30%) if resuming, unless explicitly overridden
            if eps_start == 1.0:
                eps_start = 0.30
                print(f"[Resume] Set initial epsilon exploration to {eps_start:.2f} (prevents wiping learned policy).")
        else:
            print(f"[Warning] Checkpoint '{resume_from}' does not exist! Starting fresh from scratch.")

    try:
        for episode in range(start_episode, start_episode + num_episodes):
            state = format_state(env.reset())
            done = False
            episode_reward = 0.0
            episode_losses = []
            episode_firing_rates = []

            # Linear epsilon decay relative to remaining training episodes
            progress = min(1.0, (episode - start_episode) / float(max(1, eps_decay_episodes)))
            epsilon = eps_start - progress * (eps_start - eps_end)

            while not done:
                global_step += 1
                action, step_fr = select_action(policy_net, state, epsilon, num_actions, device)

                # Send action to ViZDoom
                viz_action = to_vizdoom_action(env, action)
                next_raw_state, reward, done, _ = env.step(viz_action)

                next_state = format_state(next_raw_state) if next_raw_state is not None else np.zeros_like(state)

                # Scaled reward for gradient stability
                scaled_reward = float(reward) / 100.0
                replay_buffer.push(state, action, scaled_reward, next_state, done)

                state = next_state
                episode_reward += float(reward)
                if step_fr > 0.0:
                    episode_firing_rates.append(step_fr)

                if len(replay_buffer) >= max(warmup_steps, batch_size) and (global_step % train_freq == 0):
                    loss_val, batch_fr = optimize_snn_step(
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

            reward_window.append(episode_reward)
            avg_reward_50 = float(np.mean(reward_window))
            avg_loss = float(np.mean(episode_losses)) if episode_losses else 0.0
            avg_fr = float(np.mean(episode_firing_rates)) if episode_firing_rates else 0.0

            if len(reward_window) >= 20 and avg_reward_50 > best_avg_reward:
                best_avg_reward = avg_reward_50
                best_path = os.path.join(checkpoint_dir, "snn_policy_best.pth")
                torch.save(
                    {
                        "episode": episode,
                        "model_state_dict": policy_net.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "avg_reward_50": avg_reward_50,
                        "time_steps": time_steps,
                        "input_channels": input_channels,
                        "num_actions": num_actions,
                    },
                    best_path,
                )

            if episode % 10 == 0 or episode == 1:
                elapsed = time.time() - start_time
                betas = [
                    round(policy_net.lif1.beta.mean().item(), 3),
                    round(policy_net.lif2.beta.mean().item(), 3),
                    round(policy_net.lif3.beta.mean().item(), 3),
                    round(policy_net.lif_fc.beta.mean().item(), 3),
                ]
                print(
                    f"Ep {episode:4d}/{start_episode + num_episodes} | "
                    f"Reward: {episode_reward:6.1f} | "
                    f"Avg50: {avg_reward_50:6.1f} | "
                    f"Eps: {epsilon:.2f} | "
                    f"Loss: {avg_loss:.4f} | "
                    f"SpikeRate: {avg_fr*100:4.1f}% | "
                    f"Betas: {betas} | "
                    f"Time: {elapsed:.0f}s"
                )

        final_path = os.path.join(checkpoint_dir, "snn_policy_final.pth")
        torch.save(policy_net.state_dict(), final_path)
        print(f"Training complete! Best 50-episode average reward: {best_avg_reward:.1f}")

    finally:
        env.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train or fine-tune Spiking Dueling DQN on ViZDoom")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume training from (e.g. checkpoints/snn_policy_best.pth)")
    parser.add_argument("--episodes", type=int, default=800, help="Number of episodes to train")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate (e.g. 1e-4 or 5e-5 for fine-tuning)")
    parser.add_argument("--eps-start", type=float, default=None, help="Initial exploration epsilon (default: 1.0 for scratch, 0.30 if resuming)")
    parser.add_argument("--eps-end", type=float, default=0.08, help="Final exploration epsilon")
    parser.add_argument("--decay-episodes", type=int, default=650, help="Episodes over which epsilon decays")
    args = parser.parse_args()

    kwargs = {
        "num_episodes": args.episodes,
        "lr": args.lr,
        "eps_end": args.eps_end,
        "eps_decay_episodes": args.decay_episodes,
    }
    if args.resume:
        kwargs["resume_from"] = args.resume
    if args.eps_start is not None:
        kwargs["eps_start"] = args.eps_start

    train_snn(**kwargs)