"""Measure the sim gripper's command convention: command sweep -> fingertip gap width.

Run this before trusting any grasp number. The convention is not documented anywhere
and getting it backwards produces plausible-looking "grasps" that are artifacts. On
so101-nexus the measured answer is: -10 deg = closed, +100 deg = open (the reverse of
the obvious assumption). The script prints the gap at each commanded angle and saves
three close-up renders so you can see it with your own eyes.

Usage:
    python collect/measure_gripper.py            # prints table + writes gap_*.png
"""
import pathlib

import mujoco
import numpy as np
from PIL import Image

from so101_nexus import OverheadCamera, WristCamera
from so101_nexus.mujoco.pick_and_place import PickAndPlaceEnv, PickAndPlaceConfig
from so101_nexus.objects import CubeObject

from molmoact2_so101_sim import realism

realism.install()
cfg = PickAndPlaceConfig(
    obs_mode="visual",
    observations=list(PickAndPlaceConfig().observations or []) + [OverheadCamera(), WristCamera()],
    objects=[CubeObject(half_size=0.020, color="red")])
env = PickAndPlaceEnv(config=cfg, render_mode="rgb_array")
U = env.unwrapped
model, data = U.model, U.data
lo, hi = env.action_space.low, env.action_space.high


def tips():
    """Lowest geom of each jaw = the fingertip pair."""
    f = min((data.geom_xpos[g] for g in U._gripper_geom_ids), key=lambda p: p[2]).copy()
    j = min((data.geom_xpos[g] for g in U._jaw_geom_ids), key=lambda p: p[2]).copy()
    return f, j


env.reset(seed=5)
q0 = U._get_current_qpos()[:5].copy()
cam = mujoco.MjvCamera()
cam.type = mujoco.mjtCamera.mjCAMERA_FREE
ren = mujoco.Renderer(model, height=480, width=640)

print("cmd_deg | gap_mm | gap axis (unit)")
for cmd_deg in (-10, 0, 20, 40, 60, 80, 100):
    g = np.deg2rad(cmd_deg)
    a = np.concatenate([q0, [g]]).astype(np.float32)
    for _ in range(25):
        env.step(np.clip(a, lo, hi))
    f, j = tips()
    gap = float(np.linalg.norm(f - j))
    ax = (j - f) / (gap + 1e-9)
    print(f"  {cmd_deg:+4d}  | {gap*1000:6.1f} | {np.round(ax, 2).tolist()}")
    if cmd_deg in (-10, 40, 100):
        cam.lookat[:] = (f + j) / 2
        cam.distance = 0.16
        cam.elevation = -15
        cam.azimuth = 100
        ren.update_scene(data, camera=cam)
        Image.fromarray(ren.render()).save(f"{pathlib.Path.cwd()}/gap_{cmd_deg:+d}.png")
print("done")
