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
 - Axis/scale/FPS preprocessing is implemented as a lightweight, deterministic layer (M2). For now we **assume Axis Studio exports Y-up BVH** and we rotate it to the Z-up viewer (Axis Studio fixed preset). If your sample looks wrong, use the Flip toggles or switch presets.
- Assumes frame data can be reshaped into (Frames, total_channels). If a BVH is malformed, it raises a clear error.

## M2 (Preprocess) controls
After loading a BVH you can adjust:
 - **Axis preset** (None / Axis Studio fixed / common swaps)
- **Flip X/Y/Z** (to fix mirroring/handedness)
- **FPS resampling** (keep / 30 / 60)
- **Scale** (Auto / None / Factor)

These transforms are applied to world-space joint positions (safe for early-stage) and are designed to stabilize the next stages (classification + naturalness metrics).
