"""Calibration guards: affine round-trip, offset re-derivation, joint order."""
from __future__ import annotations

import numpy as np

from molmoact2_so101_sim.adapter import DEFAULT_JOINT_SIGNS, apply_calibration
from molmoact2_so101_sim.calibration import (
    SO101_JOINTS,
    joint_order_matches,
    rederive_offsets,
    round_trip_ok,
)


def test_round_trip_ok_for_baked_constants():
    assert round_trip_ok() is True


def test_round_trip_fails_on_non_invertible_zero_sign():
    assert round_trip_ok(signs=[0, 1, 1, 1, 1, 1]) is False


def test_rederive_offsets_recovers_known_offset():
    sim_home = np.array([5, -10, 15, 20, -25, 30], np.float32)
    known = np.array([2, 88, 92, 1, -3, 4], np.float32)
    ckpt_home = apply_calibration(sim_home, DEFAULT_JOINT_SIGNS, known)
    recovered = rederive_offsets(sim_home, ckpt_home, DEFAULT_JOINT_SIGNS)
    assert np.allclose(recovered, known, atol=1e-3)


def test_rederive_offsets_defaults_to_baked_signs():
    sim_home = np.zeros(6, np.float32)
    known = np.array([1, 2, 3, 4, 5, 6], np.float32)
    ckpt_home = apply_calibration(sim_home, DEFAULT_JOINT_SIGNS, known)
    recovered = rederive_offsets(sim_home, ckpt_home)   # signs=None -> baked defaults
    assert np.allclose(recovered, known, atol=1e-3)


def test_joint_order_matches_identical_sequences():
    assert joint_order_matches(SO101_JOINTS, SO101_JOINTS) is True
    assert joint_order_matches(list(SO101_JOINTS), list(SO101_JOINTS)) is True


def test_joint_order_detects_reorder():
    swapped = list(SO101_JOINTS)
    swapped[0], swapped[1] = swapped[1], swapped[0]
    assert joint_order_matches(swapped, SO101_JOINTS) is False


def test_joint_order_coerces_to_str():
    assert joint_order_matches([1, 2, 3], ["1", "2", "3"]) is True
    assert joint_order_matches([1, 2, 3], ["1", "3", "2"]) is False
