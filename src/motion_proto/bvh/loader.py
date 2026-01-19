from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from .types import BVHMotion, BVHNode


class BVHParseError(RuntimeError):
    pass


def _tokenize_lines(text: str) -> List[List[str]]:
    lines: List[List[str]] = []
    for raw in text.splitlines():
        s = raw.strip()
        if not s:
            continue
        # BVH does not officially support comments; if present, strip // style
        if "//" in s:
            s = s.split("//", 1)[0].strip()
            if not s:
                continue
        lines.append(s.split())
    return lines


@dataclass
class _Cursor:
    lines: List[List[str]]
    i: int = 0

    def peek(self) -> List[str]:
        if self.i >= len(self.lines):
            return []
        return self.lines[self.i]

    def pop(self) -> List[str]:
        if self.i >= len(self.lines):
            raise BVHParseError("Unexpected EOF")
        t = self.lines[self.i]
        self.i += 1
        return t

    def expect(self, word: str) -> None:
        t = self.pop()
        if not t or t[0] != word:
            raise BVHParseError(f"Expected '{word}', got: {' '.join(t) if t else 'EOF'}")


def _parse_node(cur: _Cursor) -> BVHNode:
    head = cur.pop()
    if not head:
        raise BVHParseError("Unexpected EOF while reading node")

    if head[0] == "End" and len(head) >= 2 and head[1] == "Site":
        node = BVHNode(name="EndSite", is_end_site=True)
    elif head[0] in ("ROOT", "JOINT") and len(head) >= 2:
        node = BVHNode(name=head[1])
    else:
        raise BVHParseError(f"Unexpected node header: {' '.join(head)}")

    # '{'
    brace = cur.pop()
    if not brace or brace[0] != "{":
        raise BVHParseError(f"Expected '{{' after node header, got: {' '.join(brace) if brace else 'EOF'}")

    while True:
        t = cur.peek()
        if not t:
            raise BVHParseError("Unexpected EOF inside node")
        if t[0] == "}":
            cur.pop()
            break

        key = t[0]
        if key == "OFFSET":
            parts = cur.pop()
            if len(parts) != 4:
                raise BVHParseError(f"OFFSET must have 3 values, got: {' '.join(parts)}")
            node.offset = (float(parts[1]), float(parts[2]), float(parts[3]))
        elif key == "CHANNELS":
            parts = cur.pop()
            if len(parts) < 3:
                raise BVHParseError(f"Bad CHANNELS line: {' '.join(parts)}")
            n = int(parts[1])
            chs = parts[2:]
            if len(chs) != n:
                raise BVHParseError(f"CHANNELS count mismatch: expected {n}, got {len(chs)}")
            node.channels = chs
        elif key in ("JOINT", "End"):
            child = _parse_node(cur)
            # name end sites uniquely by parent
            if child.is_end_site:
                child.name = f"{node.name}_EndSite"
            node.children.append(child)
        else:
            raise BVHParseError(f"Unknown key in HIERARCHY: {' '.join(t)}")

    return node


def _collect_channel_map(root: BVHNode) -> Dict[int, List[int]]:
    """Return node->channel indices in the exact CHANNELS appearance order (preorder traversal)."""
    mapping: Dict[int, List[int]] = {}
    idx = 0

    def rec(node: BVHNode) -> None:
        nonlocal idx
        if node.channels:
            mapping[id(node)] = list(range(idx, idx + len(node.channels)))
            idx += len(node.channels)
        for ch in node.children:
            rec(ch)

    rec(root)
    return mapping


def load_bvh(path: str) -> BVHMotion:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    lines = _tokenize_lines(text)
    cur = _Cursor(lines)

    # HIERARCHY
    cur.expect("HIERARCHY")
    # ROOT ...
    root = _parse_node(cur)

    # MOTION
    cur.expect("MOTION")

    # Frames: N
    frames_line = cur.pop()
    if len(frames_line) < 2 or not frames_line[0].startswith("Frames"):
        raise BVHParseError(f"Expected 'Frames: N', got: {' '.join(frames_line)}")
    # handle both 'Frames:' and 'Frames:' 'N'
    if frames_line[0] == "Frames:" and len(frames_line) >= 2:
        frames = int(frames_line[1])
    else:
        # possibly split like ['Frames:', '100'] anyway
        frames = int(frames_line[-1].replace(":", "")) if frames_line[-1].isdigit() else int(frames_line[1])

    # Frame Time: SPF
    ft_line = cur.pop()
    if len(ft_line) < 3 or ft_line[0] != "Frame" or ft_line[1] != "Time:":
        # sometimes it's ['Frame', 'Time:', '0.0333333']
        raise BVHParseError(f"Expected 'Frame Time: <sec>', got: {' '.join(ft_line)}")
    frame_time = float(ft_line[2])

    channel_map = _collect_channel_map(root)
    total_channels = 0
    for idxs in channel_map.values():
        total_channels = max(total_channels, max(idxs) + 1)

    # Collect numeric tokens for motion frames
    nums: List[float] = []
    while cur.i < len(cur.lines):
        toks = cur.pop()
        for x in toks:
            nums.append(float(x))

    expected = frames * total_channels
    if expected == 0:
        raise BVHParseError("No motion channels found (total_channels=0)")
    if len(nums) < expected:
        raise BVHParseError(f"MOTION data is too short: got {len(nums)} floats, expected {expected}")
    if len(nums) > expected:
        # allow extra trailing values but warn by truncation behavior
        nums = nums[:expected]

    data: List[List[float]] = []
    for fidx in range(frames):
        row = nums[fidx * total_channels : (fidx + 1) * total_channels]
        data.append(row)

    return BVHMotion(
        root=root,
        frames=frames,
        frame_time=frame_time,
        data=data,
        channel_index_by_node=channel_map,
    )
