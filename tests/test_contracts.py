"""Observation.validate accepts the contract and rejects malformed fields."""
from __future__ import annotations

import numpy as np
import pytest

from molmoact2_so101_sim.contracts import Observation


def _obs(cam0=None, cam1=None, state=None, task="t") -> Observation:
    return Observation(
        cam0=np.zeros((4, 4, 3), np.uint8) if cam0 is None else cam0,
        cam1=np.zeros((4, 4, 3), np.uint8) if cam1 is None else cam1,
        state=np.zeros(6, np.float32) if state is None else state,
        task=task,
    )


def test_validate_passes_on_good_observation():
    assert _obs().validate() is None


@pytest.mark.parametrize("bad", [
    {"cam0": np.zeros((4, 4, 3), np.float32)},   # wrong dtype
    {"cam0": np.zeros((4, 4), np.uint8)},        # wrong ndim
    {"cam0": np.zeros((4, 4, 4), np.uint8)},     # wrong channel count
    {"cam1": np.zeros((4, 4, 1), np.uint8)},     # second camera also checked
    {"state": np.zeros(5, np.float32)},          # wrong state shape
    {"state": np.zeros(7, np.float32)},
])
def test_validate_raises_on_bad_fields(bad):
    with pytest.raises(ValueError):
        _obs(**bad).validate()
