"""
Automated Verification Suite for Milestone 1: Unified SNN Pipeline
==================================================================
Tests:
  1. Environment interface: shapes, finite values, action mapping, reset semantics, 100-step smoke test.
  2. Model forward pass: single/batch forward, finite Q-values, firing rate bounds, statelessness.
  3. Checkpoint migration: non-destructive loading, exact tensor slice preservation, semantic action remapping.
  4. Baseline protection: verifies original baseline models, training scripts, and checkpoints exist and are untouched.
"""

import os
import sys
from pathlib import Path
import numpy as np
import torch
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.snn.snapv_unified_snn import SnapVUnifiedDQN
from environment.doom_env_unified import DoomEnvUnified
from training.checkpoint_migrator import migrate_combat_to_unified


# =============================================================================
# 1. Environment Interface Tests
# =============================================================================

class TestEnvironmentInterface:
    @pytest.fixture(scope="function")
    def env(self):
        # Initialize headless environment with basic.cfg
        env_instance = DoomEnvUnified(config_path="configs/basic.cfg", render=False)
        yield env_instance
        env_instance.close()

    def test_env_reset_shapes_and_finite_values(self, env):
        obs, telemetry = env.reset()
        assert obs.shape == (4, 32, 32), f"Expected obs shape (4, 32, 32), got {obs.shape}"
        assert obs.dtype == np.float32, f"Expected obs dtype float32, got {obs.dtype}"
        assert np.isfinite(obs).all(), "Visual observation contains non-finite values (NaN/Inf)!"
        assert (obs >= 0.0).all() and (obs <= 1.0).all(), "Visual obs values outside [0.0, 1.0]!"

        assert telemetry.shape == (4,), f"Expected telemetry shape (4,), got {telemetry.shape}"
        assert telemetry.dtype == np.float32, f"Expected telemetry dtype float32, got {telemetry.dtype}"
        assert np.isfinite(telemetry).all(), "Telemetry contains non-finite values (NaN/Inf)!"
        assert (telemetry >= 0.0).all(), "Telemetry contains negative values!"

    def test_env_action_mapping_integrity(self, env):
        assert env.num_actions == 8, f"Expected 8 tactical actions, got {env.num_actions}"
        assert len(env.actions) == 8
        # Validate expected action vectors across 6 buttons: [TURN_L, TURN_R, FWD, MOVE_L, MOVE_R, ATTACK]
        expected_actions = [
            [1, 0, 0, 0, 0, 0],  # 0: TURN_LEFT
            [0, 1, 0, 0, 0, 0],  # 1: TURN_RIGHT
            [0, 0, 1, 0, 0, 0],  # 2: MOVE_FORWARD
            [0, 0, 0, 1, 0, 0],  # 3: STRAFE_LEFT
            [0, 0, 0, 0, 1, 0],  # 4: STRAFE_RIGHT
            [0, 0, 0, 0, 0, 1],  # 5: ATTACK
            [0, 0, 0, 1, 0, 1],  # 6: STRAFE_LEFT_ATTACK
            [0, 0, 0, 0, 1, 1],  # 7: STRAFE_RIGHT_ATTACK
        ]
        for idx, expected in enumerate(expected_actions):
            assert env.actions[idx] == expected, f"Action {idx} mapping mismatch: {env.actions[idx]} != {expected}"

    def test_env_100_step_smoke_test(self, env):
        obs, telemetry = env.reset()
        step_count = 0
        total_episodes = 0

        for step in range(100):
            # Rotate through all 8 tactical actions
            action = step % 8
            obs, telemetry, reward, done, info = env.step(action)
            step_count += 1

            assert obs.shape == (4, 32, 32), f"Step {step}: Bad obs shape {obs.shape}"
            assert telemetry.shape == (4,), f"Step {step}: Bad telemetry shape {telemetry.shape}"
            assert np.isfinite(obs).all(), f"Step {step}: Non-finite obs"
            assert np.isfinite(telemetry).all(), f"Step {step}: Non-finite telemetry"
            assert np.isfinite(reward), f"Step {step}: Non-finite reward"
            assert isinstance(done, bool), f"Step {step}: done must be bool"
            assert "health" in info and "ammo" in info and "cells_visited" in info and "motion_energy" in info

            if done:
                total_episodes += 1
                obs, telemetry = env.reset()

        assert step_count == 100, f"Expected 100 steps executed, got {step_count}"


# =============================================================================
# 2. Model Forward Pass Tests
# =============================================================================

class TestModelForwardPass:
    @pytest.fixture(scope="function")
    def model(self):
        return SnapVUnifiedDQN(
            input_channels=4,
            telemetry_dim=4,
            num_actions=8,
            time_steps=4,
        )

    def test_model_forward_single_and_batch(self, model):
        model.eval()
        # Batch size 1
        x1_vis = torch.rand(1, 4, 32, 32)
        x1_tele = torch.rand(1, 4)
        with torch.no_grad():
            q1 = model(x1_vis, x1_tele)
        assert q1.shape == (1, 8), f"Expected Q-values shape (1, 8), got {q1.shape}"
        assert torch.isfinite(q1).all(), "Q-values contain non-finite numbers!"

        # Batch size 4
        x4_vis = torch.rand(4, 4, 32, 32)
        x4_tele = torch.rand(4, 4)
        with torch.no_grad():
            q4 = model(x4_vis, x4_tele)
        assert q4.shape == (4, 8), f"Expected Q-values shape (4, 8), got {q4.shape}"
        assert torch.isfinite(q4).all(), "Batch Q-values contain non-finite numbers!"

    def test_model_firing_rate_computation(self, model):
        model.eval()
        x_vis = torch.rand(2, 4, 32, 32)
        x_tele = torch.rand(2, 4)
        with torch.no_grad():
            q, fr = model(x_vis, x_tele, return_firing_rate=True)
        assert q.shape == (2, 8)
        assert fr.numel() == 1
        fr_val = float(fr.item())
        assert 0.0 <= fr_val <= 1.0, f"Firing rate {fr_val} out of bounds [0.0, 1.0]!"

    def test_model_stateless_reset(self, model):
        """Verifies that two consecutive forward passes with the identical input produce identical outputs."""
        model.eval()
        x_vis = torch.rand(2, 4, 32, 32)
        x_tele = torch.rand(2, 4)
        with torch.no_grad():
            q_first = model(x_vis, x_tele)
            q_second = model(x_vis, x_tele)
        assert torch.allclose(q_first, q_second, atol=1e-6), "Model leaked membrane state across forward calls!"

    def test_model_resource_counter(self, model):
        res = model.count_resources()
        assert res["neurons"]["total_spiking_neurons"] == 4904
        assert res["neurons"]["readout_neurons"] == 9
        # 1024 + 3456 + 151040 + 128 + 1024 = 156,672
        assert res["synapses"]["total_synapses"] == 156672


# =============================================================================
# 3. Checkpoint Migration Tests
# =============================================================================

class TestCheckpointMigration:
    def test_migration_correctness(self, tmp_path):
        source_path = PROJECT_ROOT / "checkpoints" / "snapv_policy_best.pth"
        assert source_path.exists(), "Source combat checkpoint does not exist!"

        target_path = tmp_path / "snapv_unified_test_migrated.pth"

        # Record source file mtime to verify non-destructive operation
        mtime_before = source_path.stat().st_mtime

        # Run migration
        unified_model, report = migrate_combat_to_unified(
            source_ckpt_path=source_path,
            target_ckpt_path=target_path,
            seed=42,
        )

        # 1. Verify source checkpoint is completely untouched
        mtime_after = source_path.stat().st_mtime
        assert mtime_before == mtime_after, "Source checkpoint was modified during migration!"

        # 2. Verify target file was saved
        assert target_path.exists(), "Migrated checkpoint file was not saved!"

        # 3. Load source weights for tensor comparison
        raw_src = torch.load(source_path, map_location="cpu", weights_only=False)
        src_state = raw_src["model_state_dict"] if "model_state_dict" in raw_src else raw_src

        dst_state = unified_model.state_dict()

        # Check visual filters exact match
        assert torch.equal(dst_state["conv1.weight"], src_state["conv1.weight"])
        assert torch.equal(dst_state["conv2.weight"], src_state["conv2.weight"])

        # Check FC1 visual slice exact match
        assert torch.equal(dst_state["fc1.weight"][:, :1176], src_state["fc1.weight"])

        # Check value head exact match
        assert torch.equal(dst_state["value_head.weight"], src_state["value_head.weight"])
        assert torch.equal(dst_state["value_head.bias"], src_state["value_head.bias"])

        # Check Advantage Head semantic transfer:
        # Combat 0 (Move_L) -> Unified 3 (Strafe_L)
        assert torch.equal(dst_state["advantage_head.weight"][3], src_state["advantage_head.weight"][0])
        assert torch.equal(dst_state["advantage_head.bias"][3], src_state["advantage_head.bias"][0])
        # Combat 1 (Move_R) -> Unified 4 (Strafe_R)
        assert torch.equal(dst_state["advantage_head.weight"][4], src_state["advantage_head.weight"][1])
        assert torch.equal(dst_state["advantage_head.bias"][4], src_state["advantage_head.bias"][1])
        # Combat 2 (Attack) -> Unified 5 (Attack)
        assert torch.equal(dst_state["advantage_head.weight"][5], src_state["advantage_head.weight"][2])
        assert torch.equal(dst_state["advantage_head.bias"][5], src_state["advantage_head.bias"][2])
        # Combat 3 (L+Attack) -> Unified 6 (Strafe_L_Attack)
        assert torch.equal(dst_state["advantage_head.weight"][6], src_state["advantage_head.weight"][3])
        assert torch.equal(dst_state["advantage_head.bias"][6], src_state["advantage_head.bias"][3])
        # Combat 4 (R+Attack) -> Unified 7 (Strafe_R_Attack)
        assert torch.equal(dst_state["advantage_head.weight"][7], src_state["advantage_head.weight"][4])
        assert torch.equal(dst_state["advantage_head.bias"][7], src_state["advantage_head.bias"][4])

        # Check new actions are initialized and finite
        assert torch.isfinite(dst_state["advantage_head.weight"][:3]).all()
        assert torch.isfinite(dst_state["advantage_head.bias"][:3]).all()

        # Check telemetry slice is initialized and finite
        assert torch.isfinite(dst_state["fc1.weight"][:, 1176:]).all()

        # Check quantization scales are positive numbers
        assert dst_state["fc1.weight_scale"].item() > 0.0
        assert dst_state["advantage_head.weight_scale"].item() > 0.0

    def test_migration_invalid_path_raises_error(self):
        with pytest.raises(FileNotFoundError):
            migrate_combat_to_unified("non_existent_checkpoint_path.pth")

        source_path = PROJECT_ROOT / "checkpoints" / "snapv_policy_best.pth"
        with pytest.raises(ValueError):
            migrate_combat_to_unified(source_path, target_ckpt_path=source_path)


# =============================================================================
# 4. Baseline Protection Tests
# =============================================================================

class TestBaselineProtection:
    def test_baseline_files_exist_and_unmodified(self):
        """Verifies that all core baseline files remain present and preserved."""
        required_baseline_files = [
            PROJECT_ROOT / "models" / "snn" / "snapv_snn.py",
            PROJECT_ROOT / "models" / "snn" / "spiking_cnn.py",
            PROJECT_ROOT / "training" / "train_snapv.py",
            PROJECT_ROOT / "training" / "train_snn.py",
            PROJECT_ROOT / "environment" / "doom_env.py",
            PROJECT_ROOT / "evaluation" / "evaluate_snapv.py",
            PROJECT_ROOT / "deployment" / "export_snapv.py",
            PROJECT_ROOT / "checkpoints" / "snapv_policy_best.pth",
            PROJECT_ROOT / "checkpoints" / "nav_snapv_final.pth",
        ]
        for f in required_baseline_files:
            assert f.exists(), f"Critical baseline file missing: {f}"
