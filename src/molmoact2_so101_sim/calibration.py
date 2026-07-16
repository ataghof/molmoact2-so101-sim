"""Calibration guards: verify the baked real-arm calibration actually holds *in sim*,
and re-derive it when it does not.

WHY THIS FILE EXISTS. MolmoAct2-SO100_101 speaks a joint convention fit to the *real*
SO-101's servo zeros and rotation senses. The `-LeRobot` checkpoint bakes that calibration
(`signs=[1,-1,1,1,1,1]`, `offsets=[0,90,90,0,0,0]`) straight into its processor. But a
simulator is a *different body*: its MJCF picks its own joint order, its own zero-pose, its
own positive-rotation sense, and radians-vs-degrees. Nothing forces those to agree with the
real arm. If even one joint's sign or zero differs, the sim arm mirrors or offsets with no
error raised, and MolmoAct2 reads as "zero-shot fails" when the fault is actually in the
bridge, not the model. A mirrored arm and a model that cannot do the task produce the same
success rate, so this module checks the bridge before any success number is trusted.

    sim.obs.state (SIM frame)   --apply_calibration-->  policy sees (CKPT frame)
                                                             |  select_action
    sim.step(target, SIM frame) <-invert_calibration--  policy action (CKPT frame)
        |
        +-> MJCF joint order / zero-pose / sign / units may NOT match the baked constants

What is here (the pure functions are deterministic and CPU-unit-testable):
    round_trip_ok       invert(apply(x)) == x for given constants (a math self-check).
    rederive_offsets    solve offsets so a sim home-pose maps onto the checkpoint home-pose.
    joint_order_matches sim joint order == checkpoint joint order.
    verify_in_sim       LIVE: nudge known joints through a real SimClient, confirm the arm
                        responds and points the expected way. (Real-sim only, not CI.)

CAUTION: a joint-space round-trip alone can NOT catch a mirrored arm. The calibration
cancels itself out: invert(apply(x)) through a perfect sim returns the input regardless of
the constants. Catching a flipped *sign* needs external ground truth about the physical arm
(the end-effector direction). That is what `JointProbe.expect_ee_*` is for, and why a
calibration check should also include an oracle positive control and a human review of the
recorded motion.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from molmoact2_so101_sim.adapter import (
    DEFAULT_JOINT_OFFSETS,
    DEFAULT_JOINT_SIGNS,
    apply_calibration,
    invert_calibration,
)
from molmoact2_so101_sim.contracts import SimClient

# Canonical SO-101 joint order (5 arm + gripper); informational labels for probe reports.
SO101_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")


# --------------------------------------------------------------------------- pure (CI-safe)


def round_trip_ok(signs=None, offsets=None, n: int = 64, atol: float = 1e-4, seed: int = 0) -> bool:
    """True iff `invert_calibration(apply_calibration(x)) ≈ x` for `n` random joint vectors.

    A self-consistency check on the affine transform: it proves apply/invert are exact
    inverses for these constants (catches a broken transform or a zero sign). It does NOT
    prove the constants are *physically correct* in sim; use `verify_in_sim` for that.
    Pure + deterministic given `seed`.
    """
    signs, offsets = _resolve(signs, offsets)
    if np.any(signs == 0.0):
        return False  # a zero sign is non-invertible (invert divides by it)
    rng = np.random.default_rng(seed)
    x = rng.uniform(-180.0, 180.0, size=(int(n), signs.size)).astype(np.float32)
    back = invert_calibration(apply_calibration(x, signs, offsets), signs, offsets)
    return bool(np.allclose(x, back, atol=atol))


def rederive_offsets(sim_home_joints, checkpoint_home_joints, signs=None) -> np.ndarray:
    """Re-fit the offsets so the sim's home-pose maps onto the checkpoint's home-pose.

    Solves `apply_calibration(sim_home, signs, offsets) == checkpoint_home` for offsets, i.e.
    `offsets = checkpoint_home_joints - sim_home_joints * signs`. This is the fix when a
    check finds the baked offsets don't hold in sim (different MJCF zero-pose). Pure; signs
    are assumed correct; a wrong *sign* is a direction bug `verify_in_sim` must flag first.
    """
    signs = DEFAULT_JOINT_SIGNS if signs is None else np.asarray(signs, np.float32)
    sim_home = np.asarray(sim_home_joints, np.float32).reshape(-1)
    ckpt_home = np.asarray(checkpoint_home_joints, np.float32).reshape(-1)
    return (ckpt_home - sim_home * signs).astype(np.float32)


def joint_order_matches(sim_joint_names, checkpoint_joint_order) -> bool:
    """True iff the two joint-name sequences are identical *in order*.

    Order mismatch is a silent sinker: MolmoAct2's 6-D state/action is positional, so a sim
    whose MJCF lists joints in a different order feeds every joint to the wrong slot.
    """
    return list(map(str, sim_joint_names)) == list(map(str, checkpoint_joint_order))


# --------------------------------------------------------------------------- live probe (real sim)


@dataclass
class JointProbe:
    """One live calibration probe: nudge `joint` by `delta_deg` (in CHECKPOINT convention),
    let the sim settle, and check it responded in the commanded direction.

    Set `expect_ee_axis` (0=x,1=y,2=z) + `expect_ee_sign` (+1/-1) from the real rig's known
    kinematics to additionally assert the end-effector moved the physically-correct way, the
    ONLY part of the probe that can catch a mirrored (sign-flipped) arm.
    """

    joint: int
    delta_deg: float = 10.0
    settle_steps: int = 25
    min_move_deg: float = 1.0          # must move ≥ this (ckpt deg) to count as "responded"
    track_tol_deg: float = 5.0         # |observed - commanded| ≤ this => "tracked" (informational)
    expect_ee_axis: int | None = None  # EE axis this joint should visibly move (ground truth)
    expect_ee_sign: float = 0.0        # +1/-1 expected sign of EE motion; 0 => don't assert
    min_ee_move: float = 1e-3          # EE must move ≥ this (m) on that axis to count
    name: str = ""


def default_probes(delta_deg: float = 10.0, settle_steps: int = 25) -> list[JointProbe]:
    """A `+delta_deg` nudge on each of the 5 ARM joints (gripper omitted: its scale and
    convention need their own measurement; see collect/measure_gripper.py). No EE-direction
    assertion by default: fill each probe's `expect_ee_axis`/`expect_ee_sign` from the real
    rig to upgrade it into a mirrored-arm guard.
    """
    return [
        JointProbe(joint=i, delta_deg=delta_deg, settle_steps=settle_steps, name=SO101_JOINTS[i])
        for i in range(5)
    ]


def verify_in_sim(sim: SimClient, signs=None, offsets=None, probes=None) -> dict[str, Any]:
    """Drive a LIVE sim through known joint nudges and report whether the arm moves as expected.

    For each probe: reset, read the sim home-pose, command `home + delta·eⱼ` in checkpoint
    space (inverted back to a sim-frame absolute joint target), step to settle, then read the
    state + privileged EE back. Per probe it records the observed joint delta (direction +
    tracking) and the EE displacement (asserted only when the probe carries a ground-truth
    `expect_ee_*`). Returns a JSON-able report; `report["ok"]` is the AND over all probes.

    Assumes `sim` speaks the so101-nexus action convention: a 6-D **absolute joint
    target** in the sim frame. CPU-import-safe: uses only numpy + the passed-in `sim`.

    NOTE: the joint direction/tracking check validates the sim's *actuators* (dead joint,
    wrong index, clamped by limits); it passes even if the calibration is mirrored, because
    the transform cancels. The `ee_direction_ok` field is the real calibration verdict; it is
    `None` until a probe is given the rig's ground-truth EE direction.
    """
    signs, offsets = _resolve(signs, offsets)
    probes = default_probes() if probes is None else list(probes)

    home_sim = _state6(sim.reset(seed=0))
    report: dict[str, Any] = {
        "ok": True,
        "n_probes": len(probes),
        "signs": [float(s) for s in signs],
        "offsets": [float(o) for o in offsets],
        "round_trip_ok": round_trip_ok(signs, offsets),
        "sim_home": [round(float(q), 4) for q in home_sim],
        "calibrated_home": [round(float(q), 4) for q in apply_calibration(home_sim, signs, offsets)],
        "probes": [],
    }

    for p in probes:
        q0_sim = _state6(sim.reset(seed=0))
        ee0 = np.asarray(sim.privileged_state().ee_pos, np.float32).reshape(-1)
        q0_ckpt = apply_calibration(q0_sim, signs, offsets)

        target_ckpt = q0_ckpt.copy()
        target_ckpt[p.joint] += p.delta_deg
        action = invert_calibration(target_ckpt, signs, offsets).astype(np.float32)

        res = None
        for _ in range(max(1, int(p.settle_steps))):
            res = sim.step(action)
            if res.done:
                break
        q1_sim = _state6(res.obs) if res is not None else q0_sim
        ee1 = np.asarray(sim.privileged_state().ee_pos, np.float32).reshape(-1)
        q1_ckpt = apply_calibration(q1_sim, signs, offsets)

        observed = float(q1_ckpt[p.joint] - q0_ckpt[p.joint])
        responded = abs(observed) >= p.min_move_deg
        direction_ok = _same_sign(observed, p.delta_deg)
        tracked = abs(observed - p.delta_deg) <= p.track_tol_deg

        ee_move = ee1 - ee0
        if p.expect_ee_axis is not None and p.expect_ee_sign:
            ax = int(p.expect_ee_axis)
            comp = float(ee_move[ax]) if ax < ee_move.size else 0.0
            ee_direction_ok: bool | None = _same_sign(comp, p.expect_ee_sign) and abs(comp) >= p.min_ee_move
        else:
            ee_direction_ok = None

        ok = bool(responded and direction_ok and (ee_direction_ok is not False))
        report["ok"] = report["ok"] and ok
        report["probes"].append({
            "name": p.name or f"joint{p.joint}",
            "joint": int(p.joint),
            "delta_deg": float(p.delta_deg),
            "observed_deg": round(observed, 4),
            "responded": responded,
            "direction_ok": direction_ok,
            "tracked": tracked,
            "ee_move": [round(float(x), 5) for x in ee_move],
            "ee_moved_norm": round(float(np.linalg.norm(ee_move)), 5),
            "ee_direction_ok": ee_direction_ok,
            "ok": ok,
        })
    return report


# --------------------------------------------------------------------------- helpers


def _resolve(signs, offsets) -> tuple[np.ndarray, np.ndarray]:
    """Substitute the baked defaults for None + coerce to float32 1-D arrays."""
    s = DEFAULT_JOINT_SIGNS if signs is None else np.asarray(signs, np.float32).reshape(-1)
    o = DEFAULT_JOINT_OFFSETS if offsets is None else np.asarray(offsets, np.float32).reshape(-1)
    return s, o


def _state6(obs) -> np.ndarray:
    """The 6-D absolute joint state out of an Observation (sim convention)."""
    return np.asarray(obs.state, np.float32).reshape(-1)[:6]


def _same_sign(a: float, b: float) -> bool:
    """True iff a and b are both strictly positive or both strictly negative."""
    return (a > 0 and b > 0) or (a < 0 and b < 0)
