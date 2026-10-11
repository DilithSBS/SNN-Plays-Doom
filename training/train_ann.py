import sys
import os
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from environment.doom_env import DoomEnv
from models.ann.cnn import SimpleCNN
from training.replay_buffer import ReplayBuffer

# ==========================================
# Hyperparameters
# ==========================================
BATCH_SIZE = 32
GAMMA = 0.99           # Discount factor for future rewards
EPS_START = 1.0        # 100% random actions at start
EPS_END = 0.1          # Minimum 10% random actions
EPS_DECAY = 5000       # How many steps to decay epsilon over
TARGET_UPDATE = 500    # How often (in steps) to update the target network
LR = 1e-4              # Learning rate
MEMORY_SIZE = 10000    # Replay buffer capacity
NUM_EPISODES = 300     # Training episodes

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_epsilon(step):
    """Calculate epsilon for epsilon-greedy exploration."""
    eps = EPS_END + (EPS_START - EPS_END) * np.exp(-1. * step / EPS_DECAY)
    return max(EPS_END, eps)

def optimize_model(policy_net, target_net, optimizer, memory, batch_size):
    """Samples a batch and performs one step of gradient descent."""
    if len(memory) < batch_size:
        return 0.0
        
    # Sample batch
    states, actions, rewards, next_states, dones = memory.sample(batch_size)
    
    # Convert to PyTorch tensors (states are already [1, 84, 84], so [batch, 1, 84, 84])
    # Next states could have Nones if episode ended, but our buffer handles this
    state_batch = torch.tensor(states, dtype=torch.float32).to(device)
    # actions are stored as integer indices, we need them as longs
    action_batch = torch.tensor(actions, dtype=torch.long).unsqueeze(1).to(device)
    reward_batch = torch.tensor(rewards, dtype=torch.float32).to(device)
    next_state_batch = torch.tensor(next_states, dtype=torch.float32).to(device)
    done_batch = torch.tensor(dones, dtype=torch.float32).to(device)
    
    # Q(s_t, a) - the model computes Q(s_t), then we gather the columns of actions taken
    state_action_values = policy_net(state_batch).gather(1, action_batch)
    
    # V(s_{t+1}) for all next states.
    # Compute max Q value for next state from the TARGET network
    with torch.no_grad():
        next_state_values = target_net(next_state_batch).max(1)[0]
        
    # Expected Q values: reward + gamma * max(Q(s_t+1)) * (1 - done)
    # If done is 1, expected Q is just the reward.
    expected_state_action_values = reward_batch + (GAMMA * next_state_values * (1 - done_batch))
    
    # Huber loss (Smooth L1) - less sensitive to outliers than MSE
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
    
    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # Gradient clipping to prevent exploding gradients
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()
    
    return loss.item()

def train():
    print(f"Starting ANN Training on {device}...")
    env = DoomEnv(config_path="configs/basic.cfg", render=True, preprocess=True)
    n_actions = env.action_space_size
    
    # Initialize networks
    policy_net = SimpleCNN(input_channels=1, num_actions=n_actions).to(device)
    target_net = SimpleCNN(input_channels=1, num_actions=n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval() # Target net is never trained directly
    
    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    memory = ReplayBuffer(MEMORY_SIZE)
    
    steps_done = 0
    best_reward = -float('inf')  # Track the best reward
    
    for episode in range(NUM_EPISODES):
        state = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            eps_threshold = get_epsilon(steps_done)
            
            # Epsilon-greedy action selection
            if random.random() < eps_threshold:
                # Random action
                action_idx = random.randrange(n_actions)
            else:
                # Greedy action from policy network
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
                with torch.no_grad():
                    q_values = policy_net(state_tensor)
                    action_idx = q_values.argmax(dim=1).item()
                    
            # Convert action index to one-hot for ViZDoom
            action_one_hot = [0] * n_actions
            action_one_hot[action_idx] = 1
            
            # Step environment
            next_state, reward, done, info = env.step(action_one_hot)
            episode_reward += reward
            
            # If done, next_state is None. We store a blank frame to keep shapes consistent
            if done:
                store_next_state = np.zeros_like(state)
            else:
                store_next_state = next_state
                
            # Store in replay buffer
            memory.push(state, action_idx, reward, store_next_state, done)
            
            # Move to next state
            state = next_state
            
            # Perform one step of optimization
            loss = optimize_model(policy_net, target_net, optimizer, memory, BATCH_SIZE)
            
            steps_done += 1
            
            # Update target network
            if steps_done % TARGET_UPDATE == 0:
                target_net.load_state_dict(policy_net.state_dict())
                
        print(f"Episode {episode+1:3d} | Reward: {episode_reward:7.1f} | Epsilon: {eps_threshold:.2f} | Steps: {steps_done}")

        # Save the best model
        if episode_reward >= best_reward and episode_reward > 0:
            best_reward = episode_reward
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(policy_net.state_dict(), "checkpoints/ann_policy_best.pth")
            print(f"--> New best model saved with reward: {best_reward:.1f}")

    env.close()
    
    # Save the final model just in case
    torch.save(policy_net.state_dict(), "checkpoints/ann_policy_final.pth")
    print("Training complete! Best model saved to checkpoints/ann_policy_best.pth")

if __name__ == "__main__":
    train()
