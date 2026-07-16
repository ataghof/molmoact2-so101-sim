"""Data types and protocols the bridge is written against.

    Observation      what a policy sees each tick (2 cams + 6-D joint state + task string)
    Action           the 6-D absolute joint target it returns
    PrivilegedState  ground-truth scene state, for scoring and scripted experts only
    SimClient        a drivable SO-101 sim (calibration.verify_in_sim probes through it)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np

# --------------------------------------------------------------------------- obs / action


@dataclass
class Observation:
    """One control tick of what a policy sees. Mirrors the MolmoAct2-SO100_101 contract:
    two RGB cameras + a 6-D *absolute* joint state + a language instruction."""

    cam0: np.ndarray   # (H, W, 3) uint8 RGB, primary / front view
    cam1: np.ndarray   # (H, W, 3) uint8 RGB, secondary / wrist view
    state: np.ndarray  # (6,) float32, absolute joint positions (5 arm + gripper)
    task: str          # natural-language instruction

    def validate(self) -> None:
        for name, cam in (("cam0", self.cam0), ("cam1", self.cam1)):
            if cam.dtype != np.uint8 or cam.ndim != 3 or cam.shape[2] != 3:
                raise ValueError(f"{name} must be (H,W,3) uint8, got {cam.shape} {cam.dtype}")
        if np.asarray(self.state).shape != (6,):
            raise ValueError(f"state must be (6,), got {np.asarray(self.state).shape}")


# A 6-D absolute joint target (5 arm + gripper), same convention as Observation.state.
Action = np.ndarray


@dataclass
class PrivilegedState:
    """Ground-truth scene state the sim exposes for scoring and scripted experts. Never
    handed to a learned policy."""

    ee_pos: np.ndarray       # (3,) end-effector position
    object_pos: np.ndarray   # (3,) manipuland position
    goal_pos: np.ndarray     # (3,) target position
    gripper: float           # 0.0 = closed .. 1.0 = open
    holding: bool            # is the object currently grasped


@dataclass
class StepResult:
    obs: Observation
    reward: float
    done: bool
    info: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- protocols


@runtime_checkable
class SimClient(Protocol):
    """A drivable SO-101 sim. `calibration.verify_in_sim` drives its probes through this
    interface, so any backend that implements it can be checked the same way."""

    control_hz: float

    def reset(self, seed: int | None = None) -> Observation: ...
    def step(self, action: Action) -> StepResult: ...
    def privileged_state(self) -> PrivilegedState: ...
    def close(self) -> None: ...
