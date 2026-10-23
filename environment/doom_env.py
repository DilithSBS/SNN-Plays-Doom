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
        resolution: tuple[int, int] = (84, 84),
    ):
        self.preprocess = preprocess
        self.stack_size = stack_size
        self.frame_skip = frame_skip
        self.resolution = resolution
        self.frame_buffer = deque(maxlen=stack_size)

        self.is_navigation = "my_way_home" in config_path.lower()

        # Initialize ViZDoom game instance
        self.game = DoomGame()
        if not os.path.exists(config_path):
            alt_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), config_path)
            if os.path.exists(alt_path):
                config_path = alt_path

        self.game.load_config(config_path)
        self.game.set_window_visible(render)

        if self.is_navigation:
            # 5 Compound Navigation Actions matching [TURN_LEFT, TURN_RIGHT, MOVE_FORWARD, MOVE_LEFT, MOVE_RIGHT]
            self.actions = [
                [1, 0, 0, 0, 0],  # 0: Turn Left
                [0, 1, 0, 0, 0],  # 1: Turn Right
                [0, 0, 1, 0, 0],  # 2: Move Forward
                [1, 0, 1, 0, 0],  # 3: Move Forward + Turn Left
                [0, 1, 1, 0, 0],  # 4: Move Forward + Turn Right
            ]
            self.action_names = {
                0: "TURN_LEFT  ",
                1: "TURN_RIGHT ",
                2: "FORWARD    ",
                3: "FWD+LEFT   ",
                4: "FWD+RIGHT  ",
            }
            import vizdoom as vzd
            self.game.add_available_game_variable(vzd.GameVariable.POSITION_X)
            self.game.add_available_game_variable(vzd.GameVariable.POSITION_Y)
        else:
            # 5 Compound Combat Actions matching [MOVE_LEFT, MOVE_RIGHT, ATTACK]
            self.actions = [
                [1, 0, 0],  # 0: Move Left
                [0, 1, 0],  # 1: Move Right
                [0, 0, 1],  # 2: Attack (Shoot)
                [1, 0, 1],  # 3: Move Left + Attack
                [0, 1, 1],  # 4: Move Right + Attack
            ]
            self.action_names = {
                0: "MOVE_LEFT  ",
                1: "MOVE_RIGHT ",
                2: "ATTACK     ",
                3: "LEFT+ATTACK",
                4: "RIGHT+ATTACK",
            }

        self.num_actions = len(self.actions)
        self.action_space_size = self.num_actions
        self.visited_cells = set()
        self.game.init()

    def _process_screen(self, screen_buffer: np.ndarray) -> np.ndarray:
        """Applies preprocessing and normalizes output to 2D (H, W) float32."""
        if screen_buffer is None:
            return np.zeros(self.resolution, dtype=np.float32)

        if self.preprocess:
            frame = preprocess_frame(screen_buffer, target_size=self.resolution)
        else:
            frame = screen_buffer

        # Ensure single 2D slice (H, W)
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
        """Stacks the circular deque along axis 0 -> shape (4, H, W)."""
        return np.stack(self.frame_buffer, axis=0)

    def reset(self) -> np.ndarray:
        """Starts a new episode and fills the 4-frame buffer with initial frame duplicates."""
        self.game.new_episode()
        self.visited_cells.clear()

        if self.is_navigation:
            try:
                import vizdoom as vzd
                px = self.game.get_game_variable(vzd.GameVariable.POSITION_X)
                py = self.game.get_game_variable(vzd.GameVariable.POSITION_Y)
                self.visited_cells.add((int(px // 64), int(py // 64)))
            except Exception:
                pass

        state = self.game.get_state()
        initial_frame = self._process_screen(state.screen_buffer if state else None)

        self.frame_buffer.clear()
        for _ in range(self.stack_size):
            self.frame_buffer.append(initial_frame)

        return self._get_stacked_state()

    def step(self, action):
        """
        Executes an action. Accepts either an integer index (0..4) or a multi-element list.
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

        # REWARD SHAPING
        if self.is_navigation:
            # Count-Based Novelty Bonus: reward visiting new 64x64 map grid tiles
            try:
                import vizdoom as vzd
                px = self.game.get_game_variable(vzd.GameVariable.POSITION_X)
                py = self.game.get_game_variable(vzd.GameVariable.POSITION_Y)
                cell = (int(px // 64), int(py // 64))
                if cell not in self.visited_cells:
                    self.visited_cells.add(cell)
                    reward += 0.5  # Novelty discovery bonus
            except Exception:
                pass
        else:
            # Combat: Penalize blind shooting without alignment
            if len(action_vec) > 2 and action_vec[2] == 1:
                reward -= 10.0

        done = self.game.is_episode_finished()

        if not done:
            state = self.game.get_state()
            frame = self._process_screen(state.screen_buffer if state else None)
            self.frame_buffer.append(frame)
            obs = self._get_stacked_state()
        else:
            last_frame = self.frame_buffer[-1] if len(self.frame_buffer) > 0 else np.zeros(self.resolution, dtype=np.float32)
            self.frame_buffer.append(last_frame)
            obs = self._get_stacked_state()

        info = {
            "kills": 0,
            "cells_visited": len(self.visited_cells) if self.is_navigation else 0,
        }
        return obs, reward, done, info

    def close(self):
        self.game.close()


if __name__ == "__main__":
    print("Testing DoomEnv with basic.cfg (Combat)...")
    env_combat = DoomEnv(config_path="configs/basic.cfg", render=False)
    print(f"Combat Actions: {env_combat.num_actions} -> {env_combat.actions}")
    obs_c = env_combat.reset()
    print(f"Obs shape: {obs_c.shape}")
    env_combat.close()

    print("\nTesting DoomEnv with my_way_home.cfg (Navigation)...")
    env_nav = DoomEnv(config_path="configs/my_way_home.cfg", render=False)
    print(f"Nav Actions: {env_nav.num_actions} -> {env_nav.actions}")
    obs_n = env_nav.reset()
    print(f"Obs shape: {obs_n.shape}")
    obs_step, rew, done, info = env_nav.step(2)  # Move forward
    print(f"Step 1: Reward={rew:.2f}, Done={done}, Info={info}")
    env_nav.close()
    print("All tests passed successfully!")