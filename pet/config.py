"""Paths to things outside this repo: the upstream Microduck environment and the policy files.

Set them with environment variables (defaults in parentheses):
    MICRODUCK_RL_DIR       the upstream microduck_rl checkout (../microduck_rl next to this repo)
    MICRODUCK_POLICY_DIR   the official pretrained policies, huggingface pollen-robotics/microduck-policies
                           rev d5a8b55 (<MICRODUCK_RL_DIR>/policies_official)
    PET_GAIT_ONNX          the pet gait trained in this project (models/petgait_v10.onnx)
    SPIN_HOP_ONNX          the spin-hop policy trained in this project (models/spin_hop.onnx)
The upstream repo is used as is, never modified: its simulator, actuator model and observation code are
imported at runtime (body/sim.py).
"""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MICRODUCK_RL_DIR = Path(os.environ.get("MICRODUCK_RL_DIR", ROOT.parent / "microduck_rl"))
POLICY_DIR = Path(os.environ.get("MICRODUCK_POLICY_DIR", MICRODUCK_RL_DIR / "policies_official"))
PET_GAIT_ONNX = Path(os.environ.get("PET_GAIT_ONNX", ROOT / "models" / "petgait_v10.onnx"))
SPIN_HOP_ONNX = Path(os.environ.get("SPIN_HOP_ONNX", ROOT / "models" / "spin_hop.onnx"))

SCENES = "src/mjlab_microduck/robot/microduck"      # scene XML files, relative to MICRODUCK_RL_DIR
APARTMENT = f"{SCENES}/scene_apartment.xml"
