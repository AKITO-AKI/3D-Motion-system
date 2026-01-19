from __future__ import annotations

"""Preprocessing utilities (Sprint M2).

This layer is the *absorption layer* for BVH source differences (Axis Studio / external datasets).
For the prototype stage, we keep it deterministic and lightweight:

- Optional FPS resampling (linear interpolation on channels; rotation channels are unwrapped)
- Optional unit scaling (auto heuristic or manual factor)
- Optional axis transform (applied to world positions; safe for early-stage)

NOTE:
Axis/handedness conversion for *rotations* is non-trivial. For M2 we apply axis conversion to
world-space joint positions after FK, which is sufficient for visualization and for the
feature/metric pipeline we will implement next.
"""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .bvh.kinematics import eval_pose_world
from .bvh.types import BVHMotion, BVHNode


PresetDict = Dict[str, object]


def presets_dir() -> Path:
    # Works both for src-layout (repo) and installed packages.
    return Path(__file__).resolve().parent / "presets"


def load_axis_presets() -> Dict[str, PresetDict]:
    """Load axis presets from JSON.

    Returns a mapping from preset id to preset dict.
    """
    path = presets_dir() / "axis_presets.json"
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    presets: Dict[str, PresetDict] = {}
    for p in data.get("presets", []):
        pid = str(p.get("id"))
        presets[pid] = p
    return presets


def axis_matrix_from_preset(preset: PresetDict) -> np.ndarray:
    m = preset.get("matrix")
    if not isinstance(m, list) or len(m) != 3:
        return np.eye(3, dtype=np.float64)
    arr = np.array(m, dtype=np.float64)
    if arr.shape != (3, 3):
        return np.eye(3, dtype=np.float64)
    return arr


@dataclass
class PreprocessConfig:
    target_fps: Optional[float] = None  # None = keep
    scale_mode: str = "auto"  # auto|none|factor
    scale_factor: float = 1.0
    axis_preset_id: str = "none"
    flip_x: bool = False
    flip_y: bool = False
    flip_z: bool = False


@dataclass
class PreprocessState:
    config: PreprocessConfig
    axis_matrix: np.ndarray
    scale_factor: float
    motion: BVHMotion


def _flatten_channel_names(root: BVHNode) -> List[str]:
    names: List[str] = []

    def rec(n: BVHNode) -> None:
        if n.channels:
            names.extend(n.channels)
        for c in n.children:
            rec(c)

    rec(root)
    return names


def resample_motion(motion: BVHMotion, target_fps: float) -> BVHMotion:
    """Resample motion channels to target FPS.

    - Positions are linearly interpolated.
    - Rotations (Euler degrees) are unwrapped per-channel then linearly interpolated.

    This is not perfect for large rotations, but is stable and deterministic for the prototype.
    """
    if target_fps <= 0:
        return motion
    if motion.frames <= 1:
        return motion

    old_dt = float(motion.frame_time)
    old_fps = 1.0 / old_dt if old_dt > 0 else target_fps
    if abs(old_fps - target_fps) < 1e-6:
        return motion

    duration = (motion.frames - 1) * old_dt
    new_dt = 1.0 / float(target_fps)
    new_frames = int(round(duration / new_dt)) + 1
    if new_frames < 2:
        new_frames = 2

    t_old = np.linspace(0.0, duration, motion.frames)
    t_new = np.linspace(0.0, duration, new_frames)

    data_old = np.asarray(motion.data, dtype=np.float64)  # (F, C)
    C = data_old.shape[1]
    names = _flatten_channel_names(motion.root)
    if len(names) != C:
        # fallback: treat all as generic scalars
        names = ["scalar"] * C

    data_new = np.empty((new_frames, C), dtype=np.float64)

    for ci in range(C):
        y = data_old[:, ci]
        ch = names[ci]

        if "rotation" in ch.lower():
            # unwrap in radians to avoid 180/360 discontinuity
            y_rad = np.deg2rad(y)
            y_un = np.unwrap(y_rad)
            y_i = np.interp(t_new, t_old, y_un)
            y_deg = np.rad2deg(y_i)
            # wrap to [-180, 180)
            y_deg = (y_deg + 180.0) % 360.0 - 180.0
            data_new[:, ci] = y_deg
        else:
            data_new[:, ci] = np.interp(t_new, t_old, y)

    # Build a shallow copy motion with updated timing/data
    m2 = BVHMotion(
        root=motion.root,
        frames=new_frames,
        frame_time=new_dt,
        data=data_new.tolist(),
        channel_index_by_node=motion.channel_index_by_node,
    )
    return m2


def estimate_scale_factor_auto(motion: BVHMotion) -> float:
    """Heuristic unit scaling.

    We estimate the overall body size (max range over axes) from frame 0 positions.
    Then choose a factor to bring it into a reasonable visualization/feature scale.

    Returns a multiplicative factor.
    """
    try:
        pose = eval_pose_world(motion, 0, include_end_sites=False)
        pts = np.asarray(pose.positions, dtype=np.float64)
        if pts.size == 0:
            return 1.0
        ranges = pts.max(axis=0) - pts.min(axis=0)
        size = float(np.max(np.abs(ranges)))
    except Exception:
        return 1.0

    # Typical human size in BVH is often around 150-200 (cm-ish) or 1.5-2.0 (m-ish)
    if size <= 0:
        return 1.0
    if size > 1000.0:
        # likely mm
        return 0.001
    if size > 100.0:
        # likely cm
        return 0.01
    if size > 10.0:
        # plausible in dm-ish, keep
        return 0.1
    if size < 5.0:
        # likely meters already
        return 1.0
    return 1.0


def build_axis_matrix(axis_preset: np.ndarray, flip_x: bool, flip_y: bool, flip_z: bool) -> np.ndarray:
    m = np.array(axis_preset, dtype=np.float64)
    if m.shape != (3, 3):
        m = np.eye(3, dtype=np.float64)

    flip = np.eye(3, dtype=np.float64)
    if flip_x:
        flip[0, 0] = -1.0
    if flip_y:
        flip[1, 1] = -1.0
    if flip_z:
        flip[2, 2] = -1.0
    return flip @ m


def apply_world_transform(pts: np.ndarray, axis_m: np.ndarray, scale: float) -> np.ndarray:
    """Apply axis and scale to world-space points."""
    out = (pts @ axis_m.T) * float(scale)
    return out


def preprocess(motion: BVHMotion, cfg: PreprocessConfig, axis_presets: Dict[str, PresetDict]) -> PreprocessState:
    m2 = motion
    if cfg.target_fps is not None:
        m2 = resample_motion(m2, float(cfg.target_fps))

    # scale
    if cfg.scale_mode == "none":
        s = 1.0
    elif cfg.scale_mode == "factor":
        s = float(cfg.scale_factor)
    else:
        s = estimate_scale_factor_auto(m2)

    # axis
    preset = axis_presets.get(cfg.axis_preset_id) if cfg.axis_preset_id else None
    base = axis_matrix_from_preset(preset) if preset else np.eye(3, dtype=np.float64)
    axis_m = build_axis_matrix(base, cfg.flip_x, cfg.flip_y, cfg.flip_z)

    return PreprocessState(config=cfg, axis_matrix=axis_m, scale_factor=s, motion=m2)
