# Motion Understanding Proto (Phase 1A)

Standalone BVH loader + skeleton viewer (stick figure).

This prototype focuses on the BVH essentials:
- Parse BVH `HIERARCHY` and `MOTION`
- Map MOTION values to channels in **the exact CHANNELS appearance order**
- Euler rotation composition in **the exact channel order**
- Forward kinematics (world-space joint positions)
- Simple viewer (PySide6 + pyqtgraph.opengl)

BVH format notes are aligned with the provided reference PDF (HIERARCHY/MOTION structure, channel ordering, Euler degrees). 

## Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# mac/linux:
. .venv/bin/activate

pip install -r requirements.txt

# run
python -m motion_proto
```

## Known limitations (intentional for prototype)
- Viewer is a simple stick figure (no mesh)
- No axis/scale normalization yet (Axis Studio preset will be added after you provide a sample BVH)
- Assumes frame data can be reshaped into (Frames, total_channels). If a BVH is malformed, it raises a clear error.
