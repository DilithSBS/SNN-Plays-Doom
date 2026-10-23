"""
SNAP-V Checkpoint Migration Utility
===================================
Migrates weights from the mature 5-action combat baseline (snapv_policy_best.pth)
into the expanded 8-action multi-modal architecture (SnapVUnifiedDQN).

Features:
  - Exact transfer of convolutional visual filters (Conv1, Conv2).
  - Explicit slice-based transfer of FC1 visual weights [0:1176].
  - Small Gaussian initialization for new telemetry connections [1176:1180].
  - Semantic remapping of combat action heads to the unified 8-action space.
  - Safe, deterministic recomputation of quantization scales.
  - Strict shape validation and comprehensive migration reporting.
  - Non-destructive: Never modifies or overwrites the source checkpoint.
"""

import os
import sys
import copy
import argparse
from pathlib import Path
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.snn.snapv_unified_snn import SnapVUnifiedDQN, fake_quantize_int8


def migrate_combat_to_unified(
    source_ckpt_path: str | Path,
    target_ckpt_path: str | Path | None = None,
    seed: int = 42,
    device: str = "cpu",
) -> tuple[SnapVUnifiedDQN, dict]:
    """
    Migrates weights from a 5-action SnapVSpikingDQN checkpoint to a SnapVUnifiedDQN model.

    Semantic Action Mapping:
        Source Combat Action 0 (MOVE_LEFT)          -> Target Action 3 (STRAFE_LEFT)
        Source Combat Action 1 (MOVE_RIGHT)         -> Target Action 4 (STRAFE_RIGHT)
        Source Combat Action 2 (ATTACK)             -> Target Action 5 (ATTACK)
        Source Combat Action 3 (LEFT+ATTACK)        -> Target Action 6 (STRAFE_LEFT_ATTACK)
        Source Combat Action 4 (RIGHT+ATTACK)       -> Target Action 7 (STRAFE_RIGHT_ATTACK)
        New Target Action 0 (TURN_LEFT)             -> Initialized (Normal, std=0.02)
        New Target Action 1 (TURN_RIGHT)            -> Initialized (Normal, std=0.02)
        New Target Action 2 (MOVE_FORWARD)          -> Initialized (Normal, std=0.02, bias=+0.05)

    Args:
        source_ckpt_path: Path to existing snapv_policy_best.pth.
        target_ckpt_path: Optional path to save the migrated checkpoint.
        seed: Random seed for reproducible initialization of new weights.
        device: Device to load and initialize models on.

    Returns:
        unified_model: Migrated SnapVUnifiedDQN instance.
        report: Detailed migration log dictionary.
    """
    torch.manual_seed(seed)
    source_path = Path(source_ckpt_path).resolve()
    if not source_path.exists():
        raise FileNotFoundError(f"[Error] Source checkpoint not found at: {source_path}")

    if target_ckpt_path is not None:
        target_path = Path(target_ckpt_path).resolve()
        if target_path == source_path:
            raise ValueError("[Error] Target checkpoint path must not be identical to source path!")

    # 1. Load source checkpoint
    raw_ckpt = torch.load(source_path, map_location=device, weights_only=False)
    if isinstance(raw_ckpt, dict) and "model_state_dict" in raw_ckpt:
        src_state = raw_ckpt["model_state_dict"]
        source_meta = {k: v for k, v in raw_ckpt.items() if k != "model_state_dict"}
    elif isinstance(raw_ckpt, dict):
        src_state = raw_ckpt
        source_meta = {}
    else:
        raise ValueError(f"[Error] Unexpected checkpoint format: {type(raw_ckpt)}")

    # 2. Instantiate destination SnapVUnifiedDQN
    unified_model = SnapVUnifiedDQN(
        input_channels=4,
        telemetry_dim=4,
        num_actions=8,
        time_steps=4,
    ).to(device)
    dst_state = unified_model.state_dict()

    report = {
        "copied": [],
        "transformed": [],
        "initialized": [],
        "skipped": [],
    }

    # 3. Direct Transfer: Conv1 and Conv2
    direct_keys = [
        "conv1.weight",
        "conv1.weight_scale",
        "conv2.weight",
        "conv2.weight_scale",
        "value_head.weight",
        "value_head.bias",
        "value_head.weight_scale",
    ]
    for key in direct_keys:
        if key not in src_state:
            raise KeyError(f"[Error] Expected key '{key}' not found in source checkpoint!")
        
        src_tensor = src_state[key]
        dst_tensor = dst_state[key]
        if src_tensor.shape != dst_tensor.shape:
            raise ValueError(
                f"[Error] Shape mismatch on '{key}': source {src_tensor.shape} != target {dst_tensor.shape}"
            )
        
        dst_state[key].copy_(src_tensor)
        report["copied"].append({
            "key": key,
            "shape": list(src_tensor.shape),
            "status": "Exact Match Direct Copy",
        })

    # 4. Sliced Transfer: FC1 (1,176 visual features + 4 telemetry features)
    src_fc1 = src_state["fc1.weight"]
    if src_fc1.shape != (128, 1176):
        raise ValueError(f"[Error] Source fc1.weight expected shape (128, 1176), got {src_fc1.shape}")

    # Copy visual weights slice [0:1176]
    dst_state["fc1.weight"][:, :1176].copy_(src_fc1)

    # Initialize telemetry weights slice [1176:1180] with small Gaussian noise
    telemetry_init = torch.randn(128, 4, device=device) * 0.01
    dst_state["fc1.weight"][:, 1176:].copy_(telemetry_init)

    # Recompute FC1 INT8 quantization scale
    _, fc1_scale = fake_quantize_int8(dst_state["fc1.weight"])
    dst_state["fc1.weight_scale"].copy_(fc1_scale.detach())

    report["transformed"].append({
        "key": "fc1.weight",
        "source_shape": list(src_fc1.shape),
        "target_shape": list(dst_state["fc1.weight"].shape),
        "visual_slice": "0..1175 (copied from source)",
        "telemetry_slice": "1176..1179 (Gaussian init, std=0.01)",
        "recomputed_scale": float(fc1_scale.item()),
    })

    # 5. Semantic Action Head Transfer: Advantage Head (5 actions -> 8 actions)
    src_adv_w = src_state["advantage_head.weight"]  # (5, 128)
    src_adv_b = src_state["advantage_head.bias"]    # (5,)
    if src_adv_w.shape != (5, 128) or src_adv_b.shape != (5,):
        raise ValueError(
            f"[Error] Source advantage_head expected (5, 128), got weight={src_adv_w.shape}, bias={src_adv_b.shape}"
        )

    # Initialize new actions: 0 (TURN_L), 1 (TURN_R), 2 (MOVE_FWD)
    new_actions_w = torch.randn(3, 128, device=device) * 0.02
    dst_state["advantage_head.weight"][:3].copy_(new_actions_w)
    dst_state["advantage_head.bias"][:3].zero_()
    # Provide slight positive exploration bias (+0.05) to forward movement
    dst_state["advantage_head.bias"][2] = 0.05

    # Transfer existing combat actions to their semantic slots:
    # 0: Left -> 3: Strafe_Left
    # 1: Right -> 4: Strafe_Right
    # 2: Attack -> 5: Attack
    # 3: Left+Attack -> 6: Strafe_Left_Attack
    # 4: Right+Attack -> 7: Strafe_Right_Attack
    action_remap = {
        0: 3,  # MOVE_LEFT -> STRAFE_LEFT
        1: 4,  # MOVE_RIGHT -> STRAFE_RIGHT
        2: 5,  # ATTACK -> ATTACK
        3: 6,  # LEFT+ATTACK -> STRAFE_LEFT_ATTACK
        4: 7,  # RIGHT+ATTACK -> STRAFE_RIGHT_ATTACK
    }
    for src_idx, dst_idx in action_remap.items():
        dst_state["advantage_head.weight"][dst_idx].copy_(src_adv_w[src_idx])
        dst_state["advantage_head.bias"][dst_idx].copy_(src_adv_b[src_idx])

    # Recompute Advantage Head quantization scale
    _, adv_scale = fake_quantize_int8(dst_state["advantage_head.weight"])
    dst_state["advantage_head.weight_scale"].copy_(adv_scale.detach())

    report["transformed"].append({
        "key": "advantage_head.weight",
        "source_shape": list(src_adv_w.shape),
        "target_shape": list(dst_state["advantage_head.weight"].shape),
        "remapping": action_remap,
        "new_actions": [0, 1, 2],
        "recomputed_scale": float(adv_scale.item()),
    })

    # 6. Load transformed state dictionary into unified model
    unified_model.load_state_dict(dst_state, strict=True)

    # 7. Optionally save target checkpoint
    if target_ckpt_path is not None:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        migrated_checkpoint = {
            "model_state_dict": unified_model.state_dict(),
            "source_checkpoint": str(source_path),
            "input_channels": 4,
            "telemetry_dim": 4,
            "num_actions": 8,
            "time_steps": 4,
            "seed": seed,
            "migration_report": report,
        }
        torch.save(migrated_checkpoint, target_path)
        report["saved_to"] = str(target_path)

    return unified_model, report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Migrate SNAP-V combat checkpoint to 8-action unified model.")
    parser.add_argument(
        "--source",
        type=str,
        default="checkpoints/snapv_policy_best.pth",
        help="Path to source combat checkpoint",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="checkpoints/snapv_unified_warmstart.pth",
        help="Path to save migrated unified checkpoint",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for new weight initialization")
    args = parser.parse_args()

    print(f"[Migrator] Migrating '{args.source}' -> '{args.target}'...")
    model, rep = migrate_combat_to_unified(args.source, args.target, seed=args.seed)
    print("Migration successful! Summary:")
    print(f"  • Copied tensors      : {len(rep['copied'])}")
    print(f"  • Transformed tensors : {len(rep['transformed'])}")
    if "saved_to" in rep:
        print(f"  • Saved checkpoint to : {rep['saved_to']}")
