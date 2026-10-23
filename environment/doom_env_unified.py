"""
Unified ViZDoom Environment Wrapper for SNAP-V SNN Pipeline
===========================================================
Features:
  - 8-Action Tactical Action Space combining 3D rotation, movement, strafing, and strafe-firing.
  - 4-Frame Grayscale Observation Stack at (4, 32, 32) resolution.
  - 4-Dimensional Normalized Telemetry Vector:
      1. Health (clamped in [0.0, 2.0] for overheal support)
      2. Ammunition (active weapon ammo clamped in [0.0, 2.0])
      3. Attack Readiness (binary flag in {0.0, 1.0})
      4. Visual Motion Energy (L1 inter-frame pixel delta in [0.0, 1.0])
  - Isolated Training-Only Spatial Novelty Exploration Bonus (using 64x64 grid coordinates).
  - Diagnostic Telemetry Logging without leaking privileged coordinates to policy inputs.
"""

import os
import sys
from collections import deque
from pathlib import Path
import numpy as np
from vizdoom import DoomGame, Button, GameVariable

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from preprocessing.image_processing import preprocess_frame


class DoomEnvUnified:
    """
    Unified Gym-like ViZDoom wrapper for multi-modal SNN agents.
    Returns:
        obs: Visual tensor of shape (4, 32, 32) float32 in [0.0, 1.0].
        telemetry: Scalar vector of shape (4,) float32 in [0.0, 2.0].
    """
    def __init__(
        self,
        config_path: str = "configs/basic.cfg",
        render: bool = False,
        frame_skip: int = 4,
        stack_size: int = 4,
        resolution: tuple[int, int] = (32, 32),
    ):
        self.config_path = str(config_path)
        self.render = render
        self.frame_skip = frame_skip
        self.stack_size = stack_size
        self.resolution = resolution
        self.frame_buffer = deque(maxlen=stack_size)
        self.visited_cells = set()

        # Initialize ViZDoom game instance
        self.game = DoomGame()
        resolved_cfg = Path(config_path)
        if not resolved_cfg.exists():
            resolved_cfg = PROJECT_ROOT / config_path
            if not resolved_cfg.exists():
                raise FileNotFoundError(f"[Error] Configuration file not found at: {config_path}")

        self.game.load_config(str(resolved_cfg))
        self.game.set_window_visible(render)

        # -----------------------------------------------------------------
        # Configure Required Buttons for the 8 Tactical Actions
        # Hardware Button Vector: [TURN_L, TURN_R, MOVE_FWD, MOVE_L, MOVE_R, ATTACK]
        # -----------------------------------------------------------------
        self.required_buttons = [
            Button.TURN_LEFT,
            Button.TURN_RIGHT,
            Button.MOVE_FORWARD,
            Button.MOVE_LEFT,
            Button.MOVE_RIGHT,
            Button.ATTACK,
        ]
        self.game.set_available_buttons(self.required_buttons)

        # Tactical 8-Action Table
        self.actions = [
            [1, 0, 0, 0, 0, 0],  # 0: TURN_LEFT
            [0, 1, 0, 0, 0, 0],  # 1: TURN_RIGHT
            [0, 0, 1, 0, 0, 0],  # 2: MOVE_FORWARD
            [0, 0, 0, 1, 0, 0],  # 3: STRAFE_LEFT
            [0, 0, 0, 0, 1, 0],  # 4: STRAFE_RIGHT
            [0, 0, 0, 0, 0, 1],  # 5: ATTACK
            [0, 0, 0, 1, 0, 1],  # 6: STRAFE_LEFT_ATTACK
            [0, 0, 0, 0, 1, 1],  # 7: STRAFE_RIGHT_ATTACK
        ]
        self.num_actions = len(self.actions)

        self.action_names = {
            0: "TURN_LEFT         ",
            1: "TURN_RIGHT        ",
            2: "MOVE_FORWARD      ",
            3: "STRAFE_LEFT       ",
            4: "STRAFE_RIGHT      ",
            5: "ATTACK            ",
            6: "STRAFE_LEFT_ATTACK",
            7: "STRAFE_RIGHT_ATTACK",
        }

        # -----------------------------------------------------------------
        # Request Required Game Variables for Telemetry & Reward
        # -----------------------------------------------------------------
        self.requested_variables = [
            GameVariable.HEALTH,
            GameVariable.SELECTED_WEAPON_AMMO,
            GameVariable.AMMO2,
            GameVariable.AMMO0,
            GameVariable.ATTACK_READY,
            GameVariable.KILLCOUNT,
            GameVariable.POSITION_X,
            GameVariable.POSITION_Y,
        ]
        for gv in self.requested_variables:
            try:
                self.game.add_available_game_variable(gv)
            except Exception:
                pass

        self.game.init()
        self.prev_raw_frame = None

    def _process_screen(self, screen_buffer: np.ndarray | None) -> np.ndarray:
        """Applies grayscale downscaling and normalizes output to 2D (32, 32) float32 in [0, 1]."""
        if screen_buffer is None:
            return np.zeros(self.resolution, dtype=np.float32)

        frame = preprocess_frame(screen_buffer, target_size=self.resolution)
        if isinstance(frame, np.ndarray):
            if frame.ndim == 3 and frame.shape[0] == 1:
                frame = frame.squeeze(0)
            elif frame.ndim == 3 and frame.shape[-1] == 1:
                frame = frame.squeeze(-1)

        frame = frame.astype(np.float32)
        if frame.max() > 1.5:
            frame /= 255.0

        return frame

    def _get_telemetry(self, current_frame: np.ndarray) -> np.ndarray:
        """
        Extracts and normalizes the 4-dimensional telemetry vector:
            v0: health_norm      in [0.0, 2.0]
            v1: ammo_norm        in [0.0, 2.0]
            v2: attack_ready     in {0.0, 1.0}
            v3: visual_motion    in [0.0, 1.0]
        """
        # 1. Health Normalization
        try:
            raw_health = float(self.game.get_game_variable(GameVariable.HEALTH))
        except Exception:
            raw_health = 100.0
        health_norm = float(np.clip(max(0.0, raw_health) / 100.0, 0.0, 2.0))

        # 2. Ammunition Normalization (checks active weapon ammo, falls back to AMMO2/AMMO0)
        ammo_found = False
        raw_ammo = 0.0
        for ammo_var in [GameVariable.SELECTED_WEAPON_AMMO, GameVariable.AMMO2, GameVariable.AMMO0]:
            try:
                val = float(self.game.get_game_variable(ammo_var))
                if val >= 0.0:
                    raw_ammo = val
                    ammo_found = True
                    break
            except Exception:
                pass
        if not ammo_found:
            raw_ammo = 50.0  # Safe neutral assumption if game does not track ammo

        ammo_norm = float(np.clip(max(0.0, raw_ammo) / 50.0, 0.0, 2.0))

        # 3. Attack Readiness Indicator
        try:
            raw_ready = float(self.game.get_game_variable(GameVariable.ATTACK_READY))
            attack_ready = 1.0 if raw_ready > 0.5 else 0.0
        except Exception:
            attack_ready = 1.0

        # 4. Visual Motion Energy (Pixel difference between consecutive frames)
        # Note: Measures visual dynamism / scene change, not true translational distance.
        if self.prev_raw_frame is not None:
            pixel_diff = np.abs(current_frame - self.prev_raw_frame).mean()
            # Normalize so that 15% average pixel shift reaches saturation (1.0)
            motion_energy = float(np.clip(pixel_diff / 0.15, 0.0, 1.0))
        else:
            motion_energy = 0.0

        self.prev_raw_frame = current_frame.copy()

        return np.array([health_norm, ammo_norm, attack_ready, motion_energy], dtype=np.float32)

    def _get_stacked_state(self) -> np.ndarray:
        """Stacks circular buffer into shape (4, 32, 32) float32."""
        return np.stack(self.frame_buffer, axis=0)

    def reset(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Starts a new episode, clears temporal buffers and returns (obs, telemetry).
        """
        self.game.new_episode()
        self.visited_cells.clear()
        self.prev_raw_frame = None

        # Track initial training-only position if available
        try:
            px = self.game.get_game_variable(GameVariable.POSITION_X)
            py = self.game.get_game_variable(GameVariable.POSITION_Y)
            self.visited_cells.add((int(px // 64), int(py // 64)))
        except Exception:
            pass

        state = self.game.get_state()
        initial_frame = self._process_screen(state.screen_buffer if state else None)

        self.frame_buffer.clear()
        for _ in range(self.stack_size):
            self.frame_buffer.append(initial_frame)

        obs = self._get_stacked_state()
        telemetry = self._get_telemetry(initial_frame)

        return obs, telemetry

    def step(self, action: int | list) -> tuple[np.ndarray, np.ndarray, float, bool, dict]:
        """
        Executes action with frame_skip ticks.
        Returns: (obs, telemetry, reward, done, info)
        """
        if isinstance(action, (int, np.integer)):
            action_idx = int(action)
            action_vec = list(self.actions[action_idx])
        else:
            action_vec = list(action)
            action_idx = 0

        # Execute in ViZDoom engine
        engine_reward = float(self.game.make_action(action_vec, self.frame_skip))
        total_reward = engine_reward

        # -----------------------------------------------------------------
        # Training-Only Intrinsic Novelty Reward (using 64x64 grid coordinates)
        # Note: Kept strictly isolated from the observation returned to the policy!
        # -----------------------------------------------------------------
        try:
            px = self.game.get_game_variable(GameVariable.POSITION_X)
            py = self.game.get_game_variable(GameVariable.POSITION_Y)
            cell = (int(px // 64), int(py // 64))
            if cell not in self.visited_cells:
                self.visited_cells.add(cell)
                # Exploration bonus for entering a brand new 64x64 tile
                total_reward += 0.5
        except Exception:
            pass

        # Ammo discipline penalty: penalize firing when ammo is empty
        is_attack_action = action_idx in (5, 6, 7) or (len(action_vec) > 5 and action_vec[5] == 1)
        if is_attack_action:
            try:
                curr_ammo = float(self.game.get_game_variable(GameVariable.SELECTED_WEAPON_AMMO))
                if curr_ammo <= 0.0:
                    total_reward -= 2.0  # Dry firing penalty
            except Exception:
                pass

        done = self.game.is_episode_finished()

        if not done:
            state = self.game.get_state()
            frame = self._process_screen(state.screen_buffer if state else None)
            self.frame_buffer.append(frame)
        else:
            last_frame = self.frame_buffer[-1] if len(self.frame_buffer) > 0 else np.zeros(self.resolution, dtype=np.float32)
            self.frame_buffer.append(last_frame)
            frame = last_frame

        obs = self._get_stacked_state()
        telemetry = self._get_telemetry(frame)

        # Extract diagnostics (for training logs, not policy inputs)
        try:
            kills = int(self.game.get_game_variable(GameVariable.KILLCOUNT))
        except Exception:
            kills = 0

        try:
            health = float(self.game.get_game_variable(GameVariable.HEALTH))
        except Exception:
            health = 100.0

        info = {
            "kills": kills,
            "health": health,
            "ammo": float(telemetry[1] * 50.0),
            "cells_visited": len(self.visited_cells),
            "motion_energy": float(telemetry[3]),
            "raw_engine_reward": engine_reward,
        }

        return obs, telemetry, total_reward, done, info

    def close(self):
        self.game.close()
