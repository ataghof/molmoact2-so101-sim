"""CPU smoke tests for the release-critical paths that unit tests previously did not cover:
the dataset relabel (metadata completeness) and the sim construction. These do not need a GPU
or the 21 GB checkpoint. Sim-dependent checks skip cleanly when optional extras are absent.
"""
import glob
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pa = pytest.importorskip("pyarrow")

REPO = Path(__file__).resolve().parents[1]


def _stats(a):
    a = np.asarray(a, np.float64)
    out = {k: getattr(a, k)(0).tolist() for k in ("min", "max", "mean", "std")}
    out["count"] = [len(a)]
    out.update({f"q{q:02d}": np.quantile(a, q / 100.0, 0).tolist() for q in (1, 10, 50, 90, 99)})
    return out


def _build_joint_dataset(root: Path):
    """A minimal LeRobot-v3-shaped dataset: data Parquet + global stats + per-episode stats,
    with a CONTINUOUS gripper channel (index 5) so the binary relabel visibly changes it."""
    rows = []
    for ei in (0, 1):
        for k in range(5 + ei):
            grip = 10.0 + k * 8.0  # spans 22.5, so binary produces both 0 and 45
            rows.append({"episode_index": ei,
                         "action": np.array([1., 2., 3., 4., 5., grip], np.float32),
                         "observation.state": np.array([1., 2., 3., 4., 5., 0.5], np.float32)})
    df = pd.DataFrame(rows)
    (root / "data" / "chunk-000").mkdir(parents=True)
    df.to_parquet(root / "data" / "chunk-000" / "file-000.parquet")

    (root / "meta").mkdir(parents=True)
    A = np.stack(df["action"].to_numpy())
    json.dump({"action": _stats(A)}, open(root / "meta" / "stats.json", "w"))

    ep_rows = []
    for ei, g in df.groupby("episode_index"):
        Ae = np.stack(g["action"].to_numpy())
        r = {"episode_index": int(ei)}
        for stat, val in _stats(Ae).items():
            r[f"stats/action/{stat}"] = val
        ep_rows.append(r)
    (root / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
    pd.DataFrame(ep_rows).to_parquet(root / "meta" / "episodes" / "chunk-000" / "file-000.parquet")


def test_binary_relabel_rewrites_per_episode_stats(tmp_path):
    """make_variants --binary must rewrite the PER-EPISODE action stats, not just the global
    ones. Regression guard against stale per-episode metadata."""
    src, dst = tmp_path / "src", tmp_path / "dst"
    _build_joint_dataset(src)
    subprocess.run([sys.executable, str(REPO / "collect" / "make_variants.py"),
                    "--src", str(src), "--dst", str(dst), "--binary"], check=True)

    ep = pd.read_parquet(glob.glob(str(dst / "meta" / "episodes" / "**" / "*.parquet"), recursive=True)[0])
    for i in range(len(ep)):
        gmin = float(np.asarray(ep.iloc[i]["stats/action/min"])[5])
        gmax = float(np.asarray(ep.iloc[i]["stats/action/max"])[5])
        assert gmin in (0.0, 45.0) and gmax in (0.0, 45.0), (
            f"per-episode gripper stat is stale: min={gmin} max={gmax} (expected binary 0/45)")

    global_stats = json.load(open(dst / "meta" / "stats.json"))["action"]
    assert set(np.round(global_stats["min"], 3)) <= {0.0, 1.0, 2.0, 3.0, 4.0, 5.0}


def test_sim_constructs_and_resets():
    """The SO-101 sim builds and resets with the observation contract the collector and the
    evaluator rely on (visual mode: joint state + two cameras). Skips if the sim extras
    (mujoco + so101-nexus) are not installed."""
    import os
    import sys
    if sys.platform == "linux":                # must be set BEFORE mujoco is imported
        os.environ.setdefault("MUJOCO_GL", "egl")
    pytest.importorskip("mujoco")
    pytest.importorskip("so101_nexus")
    from so101_nexus import OverheadCamera, WristCamera
    from so101_nexus.mujoco.pick_and_place import PickAndPlaceEnv, PickAndPlaceConfig
    from so101_nexus.objects import CubeObject
    cfg = PickAndPlaceConfig(
        obs_mode="visual",
        observations=list(PickAndPlaceConfig().observations or []) + [OverheadCamera(), WristCamera()],
        objects=[CubeObject(half_size=0.015, color="red")])
    env = PickAndPlaceEnv(config=cfg, render_mode="rgb_array")
    obs, _ = env.reset(seed=0)
    assert "state" in obs and np.asarray(obs["state"]).shape[0] == 6
    assert "overhead_camera" in obs and "wrist_camera" in obs
