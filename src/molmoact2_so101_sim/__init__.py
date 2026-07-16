"""Bridge between MolmoAct2 (LeRobot) and the so101-nexus MuJoCo SO-101 sim.

The package holds the pieces every script in this repo builds on:

    adapter      joint-calibration transform (sim frame <-> checkpoint frame),
                 per-tick clamping, observation packing
    calibration  guards that verify the calibration actually holds in a given sim
    contracts    the Observation / SimClient types the above are written against
    realism      scene tuning + camera presets that match the sim's look and
                 viewpoints to the real SO-101 rigs MolmoAct2 was trained on
"""

from molmoact2_so101_sim.adapter import (
    DEFAULT_JOINT_OFFSETS,
    DEFAULT_JOINT_SIGNS,
    apply_calibration,
    clamp_step,
    invert_calibration,
    pack_observation,
)
from molmoact2_so101_sim.contracts import Observation, PrivilegedState, SimClient, StepResult

__version__ = "1.0.0"

__all__ = [
    "DEFAULT_JOINT_OFFSETS",
    "DEFAULT_JOINT_SIGNS",
    "apply_calibration",
    "clamp_step",
    "invert_calibration",
    "pack_observation",
    "Observation",
    "PrivilegedState",
    "SimClient",
    "StepResult",
    "__version__",
]
