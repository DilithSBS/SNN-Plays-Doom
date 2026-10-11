import numpy as np
import sys
import os
# Add root to sys.path so we can import preprocessing
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vizdoom import DoomGame
from preprocessing.image_processing import preprocess_frame

class DoomEnv:
    """
    A simple wrapper around ViZDoom to make it easier to interact with
    using standard RL patterns (reset, step).
    """
    
    def __init__(self, config_path="configs/basic.cfg", render=False, preprocess=True):
        self.game = DoomGame()
        self.game.load_config(config_path)
        
        # We can hide the ViZDoom window during training/headless mode
        self.game.set_window_visible(render)
        
        self.game.init()
        
        # Basic.cfg has 3 available buttons: MOVE_LEFT, MOVE_RIGHT, ATTACK
        self.action_space_size = self.game.get_available_buttons_size()
        self.preprocess = preprocess
        
    def reset(self):
        """
        Starts a new episode and returns the initial observation.
        """
        self.game.new_episode()
        return self._get_observation()
        
    def step(self, action):
        """
        Executes an action, returns (observation, reward, done, info).
        Action should be a list/array of button states, e.g. [1, 0, 0].
        """
        # make_action advances the environment by 1 tick and returns the reward
        reward = self.game.make_action(action)
        
        # REWARD SHAPING: Penalize the agent for shooting to stop blind firing.
        # action[2] corresponds to the ATTACK button.
        if action[2] == 1:
            reward -= 5.0
            
        done = self.game.is_episode_finished()
        
        if done:
            obs = None
        else:
            obs = self._get_observation()
            
        info = {
            "kills": 0  # We can extract actual game variables here later
        }
            
        return obs, reward, done, info
        
    def _get_observation(self):
        """
        Extracts the screen buffer from the game state.
        """
        state = self.game.get_state()
        if state is None:
            return None
            
        frame = state.screen_buffer
        if self.preprocess:
            frame = preprocess_frame(frame)
            
        return frame
        
    def close(self):
        self.game.close()


if __name__ == "__main__":
    # Test the wrapper
    print("Initializing Doom Environment...")
    env = DoomEnv(config_path="configs/basic.cfg", render=False)
    
    print(f"Action space size: {env.action_space_size}")
    
    obs = env.reset()
    print(f"Observation shape: {obs.shape}")
    
    # Run a few random steps
    print("\nTaking random actions:")
    for i in range(5):
        # Create a random one-hot action vector
        action = np.zeros(env.action_space_size, dtype=int)
        action_idx = np.random.randint(0, env.action_space_size)
        action[action_idx] = 1
        
        obs, reward, done, info = env.step(action.tolist())
        print(f"Step {i+1} | Action: {action.tolist()} | Reward: {reward} | Done: {done}")
        
        if done:
            break
            
    env.close()
    print("Environment closed.")
