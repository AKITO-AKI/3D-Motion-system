from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

Vec3 = Tuple[float, float, float]


@dataclass
class BVHNode:
    name: str
    offset: Vec3 = (0.0, 0.0, 0.0)
    channels: List[str] = field(default_factory=list)
    children: List["BVHNode"] = field(default_factory=list)
    is_end_site: bool = False


@dataclass
class BVHMotion:
    root: BVHNode
    frames: int
    frame_time: float
    data: List[List[float]]  # frames x total_channels

    # Mapping: node object id -> list of channel indices in `data[frame]`
    channel_index_by_node: Dict[int, List[int]] = field(default_factory=dict)

    @property
    def total_channels(self) -> int:
        return len(self.data[0]) if self.data else 0


@dataclass
class SkeletonWorld:
    joint_names: List[str]
    parent_index: List[int]
    edges: List[Tuple[int, int]]


@dataclass
class PoseWorld:
    joint_names: List[str]
    positions: List[Vec3]
    parent_index: List[int]
