"""Bridge helpers: calibration round-trip, per-tick clamp, pack_observation."""
from __future__ import annotations

import numpy as np
import pytest

from molmoact2_so101_sim.adapter import (
    apply_calibration,
    clamp_step,
    invert_calibration,
    pack_observation,
)


def test_invert_is_left_inverse_of_apply():
    rng = np.random.default_rng(0)
    for _ in range(20):
        x = rng.uniform(-180.0, 180.0, size=6).astype(np.float32)
        assert np.allclose(invert_calibration(apply_calibration(x)), x, atol=1e-3)


def test_round_trip_with_custom_constants():
    signs = np.array([1, -1, 1, -1, 1, 1], np.float32)
    offsets = np.array([0, 45, 90, 10, -5, 0], np.float32)
    x = np.linspace(-30, 30, 6).astype(np.float32)
    back = invert_calibration(apply_calibration(x, signs, offsets), signs, offsets)
    assert np.allclose(back, x, atol=1e-3)


def test_clamp_step_caps_at_15_degrees():
    prev = np.zeros(6, np.float32)
    assert np.allclose(clamp_step(prev, np.full(6, 100.0, np.float32)), 15.0)
    assert np.allclose(clamp_step(prev, np.full(6, -100.0, np.float32)), -15.0)


def test_clamp_step_passes_small_moves_through():
    prev = np.zeros(6, np.float32)
    assert np.allclose(clamp_step(prev, np.full(6, 10.0, np.float32)), 10.0)


def test_clamp_step_honours_custom_cap():
    prev = np.zeros(6, np.float32)
    assert np.allclose(clamp_step(prev, np.full(6, 100.0, np.float32), max_deg=5.0), 5.0)


def test_clamp_step_never_exceeds_cap_for_random_targets():
    rng = np.random.default_rng(1)
    for _ in range(20):
        prev = rng.uniform(-90, 90, 6).astype(np.float32)
        target = rng.uniform(-90, 90, 6).astype(np.float32)
        out = clamp_step(prev, target, 15.0)
        assert np.max(np.abs(out - prev)) <= 15.0 + 1e-4


def test_pack_observation_builds_valid_observation():
    obs = pack_observation(
        np.zeros((8, 8, 3), np.uint8), np.zeros((8, 8, 3), np.uint8), [0.1] * 6, "task",
    )
    obs.validate()   # must not raise
    assert obs.cam0.dtype == np.uint8
    assert obs.state.shape == (6,)
    assert obs.state.dtype == np.float32
    assert obs.task == "task"


def test_pack_observation_coerces_float_images_to_uint8():
    obs = pack_observation(np.full((4, 4, 3), 0.5), np.full((4, 4, 3), 0.5), np.zeros(6), "t")
    assert obs.cam0.dtype == np.uint8
    assert obs.cam0.max() <= 255


def test_pack_observation_truncates_long_state():
    obs = pack_observation(
        np.zeros((4, 4, 3), np.uint8), np.zeros((4, 4, 3), np.uint8),
        np.arange(9, dtype=np.float32), "t",
    )
    assert obs.state.shape == (6,)


def test_pack_observation_rejects_2d_camera():
    with pytest.raises(ValueError):
        pack_observation(np.zeros((8, 8), np.uint8), np.zeros((8, 8, 3), np.uint8), [0] * 6, "t")


def test_pack_observation_rejects_short_state():
    with pytest.raises(ValueError):
        pack_observation(
            np.zeros((8, 8, 3), np.uint8), np.zeros((8, 8, 3), np.uint8), [0, 0, 0], "t",
        )
