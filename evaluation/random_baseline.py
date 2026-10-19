import sys
import os
import time
import numpy as np

# Add root to sys.path so we can import environment
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from environment.doom_env import DoomEnv

def evaluate_random_agent(episodes=5, render=False):
    """
    Runs a random agent in the Doom environment and collects basic statistics.
    """
    env = DoomEnv(config_path="configs/basic.cfg", render=render, preprocess=True)
    
    total_rewards = []
    episode_lengths = []
    
    for episode in range(episodes):
        obs = env.reset()
        done = False
        episode_reward = 0
        step_count = 0
        
        while not done:
            # Create a random one-hot action
            action = np.zeros(env.action_space_size, dtype=int)
            action[np.random.randint(0, env.action_space_size)] = 1
            
            obs, reward, done, info = env.step(action.tolist())
            episode_reward += reward
            step_count += 1
            
            if render:
                time.sleep(0.028)  # Rough approximation of 35 FPS
                
        total_rewards.append(episode_reward)
        episode_lengths.append(step_count)
        
        print(f"Episode {episode + 1} | Reward: {episode_reward:7.1f} | Length: {step_count} steps")
        
    env.close()
    
    print("\n--- Baseline Results ---")
    print(f"Average Reward: {np.mean(total_rewards):.1f} ± {np.std(total_rewards):.1f}")
    print(f"Average Length: {np.mean(episode_lengths):.1f} steps")
    print("------------------------\n")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate random baseline agent in ViZDoom")
    parser.add_argument("--episodes", type=int, default=10, help="Number of baseline episodes")
    parser.add_argument("--no-render", action="store_true", help="Disable live ViZDoom window for headless/fast benchmarking")
    args = parser.parse_args()

    print("Starting Random Baseline Evaluation...\n")
    evaluate_random_agent(episodes=args.episodes, render=not args.no_render)
