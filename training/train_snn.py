import os
import sys
import time
import random
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from environment.doom_env import DoomEnv
from models.snn.spiking_cnn import SpikingDuelingDQN


class ReplayBuffer:
    """Lightweight, memory-efficient Experience Replay Buffer."""
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

        return states_t, actions_t, rewards_t, next_states_t, dones_t

    def __len__(self):
        return len(self.buffer)


def format_state(state) -> np.ndarray:
    """
    Ensures every single observation from DoomEnv has shape (C, 84, 84) as a float32/uint8 numpy array.
    """
    if isinstance(state, torch.Tensor):
        state = state.detach().cpu().numpy()
    state = np.asarray(state)

    # Strip leading batch dimension if DoomEnv returns (1, 1, 84, 84)
    if state.ndim == 4 and state.shape[0] == 1:
        state = state.squeeze(0)
    # Add channel dimension if DoomEnv returns (84, 84)
    elif state.ndim == 2:
        state = np.expand_dims(state, axis=0)
    # Permute if HWC (84, 84, C)
    elif state.ndim == 3 and state.shape[-1] in (1, 3, 4) and state.shape[0] == 84:
        state = np.transpose(state, (2, 0, 1))

    return state

def to_vizdoom_action(env, action_idx: int, num_actions: int = 3) -> list[int]:
    """
    Converts an integer action index (0, 1, 2) into a ViZDoom one-hot button list
    (e.g., [1, 0, 0], [0, 1, 0], [0, 0, 1]).
    """
    # If DoomEnv already stores an actions list (e.g. self.actions = [[1,0,0], ...])
    if hasattr(env, "actions") and isinstance(env.actions, (list, tuple, np.ndarray)):
        return list(env.actions[action_idx])
    
    # Otherwise construct the one-hot action vector
    one_hot = [0] * num_actions
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
    """
    Selects an epsilon-greedy action.
    Crucial: Places policy_net in .eval() mode so BatchNorm1d does not crash on batch_size=1.
    """
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
    """Polyak averaging for parameters and BatchNorm running buffers."""
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

    # 1. Forward pass through Online SNN (with differentiable firing rate)
    q_Online, firing_rate = policy_net(states, return_firing_rate=True)
    state_action_values = q_Online.gather(1, actions)

    # 2. Double DQN (DDQN) Target Calculation
    with torch.no_grad():
        # Online net selects best next actions
        next_actions = policy_net(next_states, return_firing_rate=False).argmax(dim=1, keepdim=True)
        # Target net evaluates those actions
        next_q_target = target_net(next_states, return_firing_rate=False).gather(1, next_actions)
        expected_q_values = rewards + (1.0 - dones) * gamma * next_q_target

    # 3. Huber Loss + Homeostatic Firing Rate Regularization
    td_loss = nn.functional.smooth_l1_loss(state_action_values, expected_q_values)
    sparsity_loss = sparsity_weight * ((firing_rate - target_firing_rate) ** 2)
    total_loss = td_loss + sparsity_loss

    # 4. Backpropagation Through Time (BPTT) with Gradient Clipping
    optimizer.zero_grad(set_to_none=True)
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=1.0)
    optimizer.step()

    # 5. Soft-update target network
    soft_update_target(policy_net, target_net, tau=0.005)

    return float(td_loss.item()), float(firing_rate.item())


def train_snn(
    num_episodes: int = 800,
    batch_size: int = 64,
    gamma: float = 0.99,
    lr: float = 1e-4,
    time_steps: int = 4,
    eps_start: float = 1.0,
    eps_end: float = 0.05,
    eps_decay_episodes: int = 450,
    warmup_steps: int = 2_000,
    train_freq: int = 4,
    target_firing_rate: float = 0.12,
    sparsity_weight: float = 0.05,
    checkpoint_dir: str = "checkpoints",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Initialize ViZDoom Environment
    env = DoomEnv(render=False)
    sample_obs = format_state(env.reset())
    input_channels = sample_obs.shape[0]
    num_actions = getattr(env, "num_actions", 3)

    print(f"[SNN-Doom] Device: {device} | Input Shape: {sample_obs.shape} | Actions: {num_actions} | T={time_steps}")

    # Initialize Online and Target Spiking Dueling DQNs
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

    try:
        for episode in range(1, num_episodes + 1):
            state = format_state(env.reset())
            done = False
            episode_reward = 0.0
            episode_losses = []
            episode_firing_rates = []

            # Linear epsilon schedule across episodes
            progress = min(1.0, episode / float(eps_decay_episodes))
            epsilon = eps_start - progress * (eps_start - eps_end)

            while not done:
                global_step += 1
                action, step_fr = select_action(policy_net, state, epsilon, num_actions, device)

                # Step ViZDoom environment (handles both 3-tuple and 4/5-tuple Gym returns)
                viz_action = to_vizdoom_action(env, action, num_actions)
                step_out = env.step(viz_action)
                if len(step_out) == 3:
                    next_raw_state, reward, done = step_out
                elif len(step_out) == 4:
                    next_raw_state, reward, done, _ = step_out
                else:
                    next_raw_state, reward, terminated, truncated, _ = step_out
                    done = terminated or truncated

                next_state = format_state(next_raw_state) if next_raw_state is not None else np.zeros_like(state)

                # Scale reward slightly if ViZDoom returns raw [-300, 100] to stabilize BPTT gradients
                scaled_reward = float(reward) / 100.0
                replay_buffer.push(state, action, scaled_reward, next_state, done)

                state = next_state
                episode_reward += float(reward)
                if step_fr > 0.0:
                    episode_firing_rates.append(step_fr)

                # Optimize SNN every `train_freq` steps once warmup is complete
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

            # Save best model checkpoint once warmup is complete
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
                    f"Ep {episode:4d}/{num_episodes} | "
                    f"Reward: {episode_reward:6.1f} | "
                    f"Avg50: {avg_reward_50:6.1f} | "
                    f"Eps: {epsilon:.2f} | "
                    f"Loss: {avg_loss:.4f} | "
                    f"SpikeRate: {avg_fr*100:4.1f}% | "
                    f"Betas: {betas} | "
                    f"Time: {elapsed:.0f}s"
                )

        # Save final checkpoint
        final_path = os.path.join(checkpoint_dir, "snn_policy_final.pth")
        torch.save(policy_net.state_dict(), final_path)
        print(f"Training complete! Best 50-ep average reward: {best_avg_reward:.1f}")

    finally:
        if hasattr(env, "close"):
            env.close()


if __name__ == "__main__":
    train_snn()