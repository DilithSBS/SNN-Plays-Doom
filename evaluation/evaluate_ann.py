import sys
import os
import time
import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from environment.doom_env import DoomEnv
from models.ann.cnn import SimpleCNN

def evaluate_model(model_path="checkpoints/ann_policy_best.pth", episodes=10, render=False):
    """
    Evaluates a trained ANN agent in the Doom environment.
    """
    print(f"Loading trained model from {model_path}...")
    
    env = DoomEnv(config_path="configs/basic.cfg", render=render, preprocess=True)
    n_actions = env.action_space_size
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = SimpleCNN(input_channels=1, num_actions=n_actions)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()  # Set to evaluation mode (no gradients, no dropout)
    
    total_rewards = []
    episode_lengths = []
    
    for episode in range(episodes):
        state = env.reset()
        done = False
        episode_reward = 0
        step_count = 0
        
        while not done:
            # Prepare state tensor
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            
            # Forward pass: get Q-values and choose action greedily (argmax)
            with torch.no_grad():
                q_values = model(state_tensor)
                action_idx = q_values.argmax(dim=1).item()
                
            # Convert to one-hot array
            action_one_hot = [0] * n_actions
            action_one_hot[action_idx] = 1
            
            # Execute action
            state, reward, done, info = env.step(action_one_hot)
            
            episode_reward += reward
            step_count += 1
            
            if render:
                time.sleep(0.028)  # ~35 FPS
                
        total_rewards.append(episode_reward)
        episode_lengths.append(step_count)
        print(f"Episode {episode + 1} | Reward: {episode_reward:7.1f} | Length: {step_count} steps")
        
    env.close()
    
    print("\n--- ANN Evaluation Results ---")
    print(f"Average Reward: {np.mean(total_rewards):.1f} ± {np.std(total_rewards):.1f}")
    print(f"Average Length: {np.mean(episode_lengths):.1f} steps")
    print("------------------------------\n")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate trained CNN (ANN) policy in ViZDoom")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/ann_policy_best.pth", help="Path to ANN checkpoint")
    parser.add_argument("--episodes", type=int, default=10, help="Number of evaluation episodes")
    parser.add_argument("--no-render", action="store_true", help="Disable live ViZDoom window for headless/fast benchmarking")
    args = parser.parse_args()

    evaluate_model(
        model_path=args.checkpoint,
        episodes=args.episodes,
        render=not args.no_render
    )
