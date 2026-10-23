import os
import sys
from collections import deque
import numpy as np
from vizdoom import DoomGame

# Add project root to sys.path so preprocessing can be resolved
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocessing.image_processing import preprocess_frame


class DoomEnv:
    """
    ViZDoom environment wrapper with 4-frame temporal stacking and compound action support.
    Returns observations of shape (4, 84, 84) as np.float32 normalized in [0.0, 1.0].
    """
    def __init__(
        self,
        config_path: str = "configs/basic.cfg",
        render: bool = False,
        preprocess: bool = True,
        stack_size: int = 4,
        frame_skip: int = 4,
    ):
        self.preprocess = preprocess
        self.stack_size = stack_size
        self.frame_skip = frame_skip
        self.frame_buffer = deque(maxlen=stack_size)

        # 5 Compound Actions matching [MOVE_LEFT, MOVE_RIGHT, ATTACK]
        self.actions = [
            [1, 0, 0],  # 0: Move Left
            [0, 1, 0],  # 1: Move Right
            [0, 0, 1],  # 2: Attack (Shoot)
            [1, 0, 1],  # 3: Move Left + Attack
            [0, 1, 1],  # 4: Move Right + Attack
        ]
        self.num_actions = len(self.actions)
        self.action_space_size = self.num_actions

        # Initialize ViZDoom game instance
        self.game = DoomGame()
        if not os.path.exists(config_path):
            alt_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), config_path)
            if os.path.exists(alt_path):
                config_path = alt_path

        self.game.load_config(config_path)
        self.game.set_window_visible(render)
        self.game.init()

    def _process_screen(self, screen_buffer: np.ndarray) -> np.ndarray:
        """Applies preprocessing and normalizes output to 2D (84, 84) float32."""
        if screen_buffer is None:
            return np.zeros((84, 84), dtype=np.float32)

        if self.preprocess:
            frame = preprocess_frame(screen_buffer)
        else:
            frame = screen_buffer

        # Ensure single 2D slice (84, 84)
        if isinstance(frame, np.ndarray):
            if frame.ndim == 3 and frame.shape[0] == 1:
                frame = frame.squeeze(0)
            elif frame.ndim == 3 and frame.shape[-1] == 1:
                frame = frame.squeeze(-1)

        frame = frame.astype(np.float32)
        if frame.max() > 1.5:
            frame /= 255.0

        return frame

    def _get_stacked_state(self) -> np.ndarray:
        """Stacks the circular deque along axis 0 -> shape (4, 84, 84)."""
        return np.stack(self.frame_buffer, axis=0)

    def reset(self) -> np.ndarray:
        """Starts a new episode and fills the 4-frame buffer with initial frame duplicates."""
        self.game.new_episode()
        state = self.game.get_state()
        initial_frame = self._process_screen(state.screen_buffer if state else None)

        self.frame_buffer.clear()
        for _ in range(self.stack_size):
            self.frame_buffer.append(initial_frame)

        return self._get_stacked_state()

    def step(self, action):
        """
        Executes an action. Accepts either an integer index (0..4) or a 3-element list.
        Returns: (obs, reward, done, info)
        """
        if isinstance(action, (int, np.integer)):
            action_vec = list(self.actions[int(action)])
        elif isinstance(action, (list, tuple, np.ndarray)):
            action_vec = list(action)
        else:
            action_vec = list(self.actions[0])

        # Execute action with frame_skip ticks
        reward = self.game.make_action(action_vec, self.frame_skip)

        # REWARD SHAPING: Penalize shooting to discourage blind firing
        if len(action_vec) > 2 and action_vec[2] == 1:
            reward -= 10.0

        done = self.game.is_episode_finished()

        if not done:
            state = self.game.get_state()
            frame = self._process_screen(state.screen_buffer if state else None)
            self.frame_buffer.append(frame)
            obs = self._get_stacked_state()
        else:
            # Duplicate latest valid frame to preserve stack structure on episode termination
            last_frame = self.frame_buffer[-1] if len(self.frame_buffer) > 0 else np.zeros((84, 84), dtype=np.float32)
            self.frame_buffer.append(last_frame)
            obs = self._get_stacked_state()

        info = {"kills": 0}
        return obs, reward, done, info

    def close(self):
        self.game.close()


if __name__ == "__main__":
    print("Testing updated DoomEnv with 4-frame stacking...")
    env = DoomEnv(render=False)
    print(f"Action space size: {env.num_actions}")
    obs = env.reset()
    print(f"Observation stack shape: {obs.shape} (Expected: (4, 84, 84))")

    for i in range(5):
        action_idx = np.random.randint(0, env.num_actions)
        obs, reward, done, _ = env.step(action_idx)
        print(f"Step {i+1} | Action {action_idx}: {env.actions[action_idx]} | Obs shape: {obs.shape} | Reward: {reward:.1f}")
        if done:
            break
    env.close()
    print("Test passed successfully.")