"""Bridge helpers between a raw sim and the MolmoAct2 contract: pack observations, apply the
joint-calibration transform, and clamp per-tick joint deltas. Pure numpy.

Verification / re-derivation of the calibration for a *specific* sim lives in
calibration.py and builds on the transforms here.
"""
from __future__ import annotations

import numpy as np

from molmoact2_so101_sim.contracts import Observation

# MolmoAct2-SO100_101's joint calibration (real-arm convention). Verify it holds in your
# sim before trusting any number; a sim's joint zeros/signs may differ. See calibration.py.
DEFAULT_JOINT_SIGNS = np.array([1, -1, 1, 1, 1, 1], dtype=np.float32)
DEFAULT_JOINT_OFFSETS = np.array([0, 90, 90, 0, 0, 0], dtype=np.float32)


def apply_calibration(sim_joints, signs=DEFAULT_JOINT_SIGNS, offsets=DEFAULT_JOINT_OFFSETS):
    """Sim joint state -> checkpoint convention:  q * signs + offsets."""
    return (np.asarray(sim_joints, np.float32) * np.asarray(signs, np.float32)
            + np.asarray(offsets, np.float32))


def invert_calibration(action, signs=DEFAULT_JOINT_SIGNS, offsets=DEFAULT_JOINT_OFFSETS):
    """Checkpoint-convention action -> sim joint target (exact inverse of apply_calibration)."""
    return (np.asarray(action, np.float32) - np.asarray(offsets, np.float32)) / np.asarray(signs, np.float32)


def clamp_step(prev, target, max_deg: float = 15.0):
    """Cap per-tick joint change to <= max_deg."""
    prev = np.asarray(prev, np.float32)
    target = np.asarray(target, np.float32)
    return prev + np.clip(target - prev, -max_deg, max_deg)


def pack_observation(cam0, cam1, state, task) -> Observation:
    """Assemble a validated Observation from raw sim outputs."""
    obs = Observation(
        cam0=_as_u8(cam0), cam1=_as_u8(cam1),
        state=np.asarray(state, np.float32).reshape(-1)[:6], task=str(task),
    )
    obs.validate()
    return obs


def _as_u8(img):
    img = np.asarray(img)
    if img.dtype == np.uint8:
        return img
    peak = float(np.nanmax(img)) if img.size else 0.0
    scale = 255.0 if peak <= 1.0 else 1.0
    return np.clip(img * scale, 0, 255).astype(np.uint8)
