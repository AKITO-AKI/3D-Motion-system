from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from .math3d import apply_transform, compose_euler_from_channels, translate
from .types import BVHMotion, BVHNode, PoseWorld, SkeletonWorld, Vec3


@dataclass
class NodeLayout:
    """Flattened node tree for FK evaluation."""

    nodes: List[BVHNode]
    parent_index: List[int]
    names: List[str]
    edges: List[Tuple[int, int]]  # parent->child


def build_layout(root: BVHNode, include_end_sites: bool = True) -> NodeLayout:
    nodes: List[BVHNode] = []
    parent_index: List[int] = []
    names: List[str] = []
    edges: List[Tuple[int, int]] = []

    def rec(node: BVHNode, pidx: int) -> None:
        idx = len(nodes)
        nodes.append(node)
        parent_index.append(pidx)
        names.append(node.name)
        if pidx >= 0:
            edges.append((pidx, idx))
        for ch in node.children:
            if (not include_end_sites) and ch.is_end_site:
                continue
            rec(ch, idx)

    rec(root, -1)
    return NodeLayout(nodes=nodes, parent_index=parent_index, names=names, edges=edges)


def get_skeleton_world(motion: BVHMotion, include_end_sites: bool = True) -> SkeletonWorld:
    layout = build_layout(motion.root, include_end_sites=include_end_sites)
    return SkeletonWorld(joint_names=layout.names, parent_index=layout.parent_index, edges=layout.edges)


def eval_pose_world(motion: BVHMotion, frame_idx: int, include_end_sites: bool = True) -> PoseWorld:
    """Forward kinematics: return world-space joint positions for a frame.

    - translation: node may have X/Y/Zposition channels (typically ROOT)
    - rotation: applied in the exact order of rotation channels in CHANNELS
    """
    if frame_idx < 0 or frame_idx >= motion.frames:
        raise IndexError(f"frame_idx out of range: {frame_idx}")

    layout = build_layout(motion.root, include_end_sites=include_end_sites)
    world: List[np.ndarray] = [np.eye(4, dtype=np.float64) for _ in layout.nodes]
    positions: List[Vec3] = []

    for idx, node in enumerate(layout.nodes):
        pidx = layout.parent_index[idx]
        parent_m = world[pidx] if pidx >= 0 else np.eye(4, dtype=np.float64)

        vals: List[float] = []
        if node.channels:
            ch_idxs = motion.channel_index_by_node.get(id(node), [])
            vals = [motion.data[frame_idx][i] for i in ch_idxs]

        pos_x = pos_y = pos_z = 0.0
        rot_channels: List[str] = []
        rot_values: List[float] = []

        for ch, v in zip(node.channels, vals):
            if ch == "Xposition":
                pos_x = v
            elif ch == "Yposition":
                pos_y = v
            elif ch == "Zposition":
                pos_z = v
            elif ch in ("Xrotation", "Yrotation", "Zrotation"):
                rot_channels.append(ch)
                rot_values.append(v)

        t = (node.offset[0] + pos_x, node.offset[1] + pos_y, node.offset[2] + pos_z)
        local = translate(t) @ compose_euler_from_channels(rot_channels, rot_values)
        world[idx] = parent_m @ local
        positions.append(apply_transform(world[idx], (0.0, 0.0, 0.0)))

    return PoseWorld(joint_names=layout.names, positions=positions, parent_index=layout.parent_index)
