"""Scripted-expert demo collector for so101-nexus pick-and-place -> LeRobot dataset.

A damped-least-squares IK expert plays the task; episodes are recorded straight into
LeRobot dataset format. No human demonstrations, no teleop. The dataset design choices
that mattered (each one is measured in the repo's experiment record):

  - DART-style noise: execute with joint wobble, record the CLEAN commands, so the
    model sees recoveries without learning the wobble.
  - Episode kinds: std / retry (engineered miss -> reopen -> real grasp) /
    midtask (starts mid-carry) / release (records only the place-and-release tail).
  - Success-state endings: demos end at the env's success signal with the
    object placed, so "done" is inside the training distribution.
  - Slow release: the gripper opens over ~11 recorded frames instead of one
    step, giving the policy a release it can actually imitate at 33 Hz.
  - Per-episode camera/light/floor jitter for cheap visual robustness.

All randomness derives from the episode seed, so a dataset regenerates deterministically.
States/actions are stored in the MolmoAct2 checkpoint convention (signs/offsets applied,
gripper mapped to 0..45), recorded at 33 Hz (every 6th step of the 200 Hz sim).

Usage:
    python collect/collect_demos.py --n 500 --root data/cube500 --repo-id local/cube500
"""
import argparse
import os
import pathlib
import shutil
import sys

if sys.platform == "linux":                    # EGL is Linux-only; macOS picks its native GL
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import mujoco
import numpy as np
from PIL import Image

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from so101_nexus import OverheadCamera, WristCamera
from so101_nexus.mujoco.pick_and_place import PickAndPlaceEnv, PickAndPlaceConfig
from so101_nexus.objects import CubeObject

from molmoact2_so101_sim import realism

HALF = 0.015                 # cube half-size (m)
SUB = 6                      # 200 Hz sim -> 33 Hz recorded
JS = np.array([1., -1., 1., 1., 1., 1.], dtype=np.float32)   # sim->checkpoint joint signs
JO = np.array([0., 90., 90., 0., 0., 0.], dtype=np.float32)  # sim->checkpoint joint offsets
NOISE_SIGMA = 0.006          # rad, DART wobble on the executed arm command

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--n", type=int, default=500, help="episodes to keep")
ap.add_argument("--root", default="data/cube500", help="dataset output directory")
ap.add_argument("--repo-id", default="local/cube500", help="LeRobot dataset repo id (metadata only)")
ap.add_argument("--task", default="Pick up the red cube and place it on the blue circle.")
ap.add_argument("--seed-cap", type=int, default=6000, help="stop after this many attempted seeds")
ap.add_argument("--overwrite", action="store_true", help="delete an existing --root first")
args = ap.parse_args()

ROOT = str(pathlib.Path(args.root).resolve())
if pathlib.Path(ROOT).exists():
    if not args.overwrite:
        sys.exit(f"{ROOT} exists; pass --overwrite to replace it")
    shutil.rmtree(ROOT)

realism.install()
cfg = PickAndPlaceConfig(
    obs_mode="visual",
    observations=list(PickAndPlaceConfig().observations or []) + [OverheadCamera(), WristCamera()],
    objects=[CubeObject(half_size=HALF, color="red")])
env = PickAndPlaceEnv(config=cfg, render_mode="rgb_array")
U = env.unwrapped
realism.tune_arm(U.model)
model, data = U.model, U.data
lo, hi = env.action_space.low, env.action_space.high
SID = U._tcp_site_id
ARM = U._arm_qvel_addrs
jacp = np.zeros((3, model.nv))
jacr = np.zeros((3, model.nv))
OPEN, CLOSED = float(hi[5]), float(lo[5])

# Grip-friendly contact params: the stock cube slips out of the stock jaws.
for gid in set(U._gripper_geom_ids) | set(U._jaw_geom_ids) | {U._obj_geom_id}:
    f = model.geom_friction[gid].copy()
    model.geom_friction[gid] = [max(f[0], 1.2), max(f[1] * 10, 0.05), max(f[2] * 10, 0.01)]

# Stash pristine light/material state so per-episode jitter never drifts.
_L0 = (model.light_pos.copy(), model.light_diffuse.copy())
_TABLE_MAT = None
for i in range(model.nmat):
    if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MATERIAL, i) or "") == "rk_table":
        _TABLE_MAT = i
_M0 = model.mat_rgba[_TABLE_MAT].copy() if _TABLE_MAT is not None else None
_FRONT = realism._CAM_PRESETS["front_low"]


def g2model(gdeg):
    """Sim gripper angle (deg, -10=closed..+100=open) -> checkpoint gripper channel (0..45)."""
    return float(np.clip((gdeg + 10.) / 110. * 45., 0, 45))


def tcp():
    return data.site_xpos[SID].copy()


def curq():
    return U._get_current_qpos()


def objp():
    return U._get_object_pose()[:3].copy()


def tgtp():
    return U._get_target_pos().copy()


def apply_randomization(rng):
    # camera: jitter the front_low pose + fov
    eye = _FRONT[0] + rng.normal(0, 0.02, 3)
    tgt = _FRONT[1] + rng.normal(0, 0.01, 3)
    fovy = _FRONT[2] + rng.normal(0, 2.0)
    d = eye - tgt
    dist = float(np.linalg.norm(d))
    cam = U._overhead_obs_cam
    cam.lookat[:] = tgt
    cam.distance = dist
    cam.elevation = float(-np.degrees(np.arcsin(d[2] / dist)))
    cam.azimuth = float(np.degrees(np.arctan2(d[1], d[0])))
    model.vis.global_.fovy = float(np.clip(fovy, 50, 66))
    # lights: position + brightness jitter around pristine values
    model.light_pos[:] = _L0[0] + rng.normal(0, 0.10, _L0[0].shape)
    model.light_diffuse[:] = np.clip(_L0[1] * rng.uniform(0.82, 1.18), 0, 1)
    # floor tint jitter
    if _TABLE_MAT is not None:
        model.mat_rgba[_TABLE_MAT] = np.clip(_M0 * np.array([*rng.uniform(0.88, 1.12, 3), 1.0]), 0, 1)


state = {"obs": None, "frames": None, "step": 0, "rng": None, "noise_on": True}


def _record(q_clean):
    if state["frames"] is None or state["step"] % SUB != 0:
        return
    obs = state["obs"]
    sd = np.rad2deg(obs["state"]).astype(np.float32)
    st = (JS * sd + JO).astype(np.float32)
    st[5] = g2model(sd[5])
    ad = np.rad2deg(q_clean).astype(np.float32)
    ac = (JS * ad + JO).astype(np.float32)
    ac[5] = g2model(ad[5])
    c0 = np.ascontiguousarray(obs["overhead_camera"]).astype(np.uint8)
    w = np.ascontiguousarray(obs["wrist_camera"]).astype(np.uint8)
    if w.shape[:2] != (480, 640):
        w = np.asarray(Image.fromarray(w).resize((640, 480))).astype(np.uint8)
    state["frames"].append((c0, w, st, ac))


def _exec(q_clean):
    _record(q_clean)
    q = q_clean.copy()
    if state["noise_on"]:
        q[:5] += state["rng"].normal(0, NOISE_SIGMA, 5)   # DART: execute noisy, record clean
    state["obs"], _r, term, _t, info = env.step(np.clip(q, lo, hi).astype(np.float32))
    state["step"] += 1
    return term, info


def ik_step(target, grip, gain=0.6, damp=0.1, maxstep=0.05):
    mujoco.mj_jacSite(model, data, jacp, jacr, SID)
    J = jacp[:, ARM]
    err = np.asarray(target, float) - tcp()
    dq = np.clip(gain * (J.T @ np.linalg.solve(J @ J.T + damp * damp * np.eye(3), err)), -maxstep, maxstep)
    q = curq().copy()
    q[:5] += dq
    q[5] = grip
    return _exec(np.clip(q, lo, hi).astype(np.float32))


def go(target, grip, steps, **kw):
    info = {}
    for _ in range(steps):
        term, info = ik_step(target, grip, **kw)
        if term:
            return info, True
    return info, False


def hold(grip, steps, noise=False):
    info = {}
    term = False
    prev = state["noise_on"]
    state["noise_on"] = noise
    for _ in range(steps):
        q = np.concatenate([curq()[:5], [grip]]).astype(np.float32)
        term, info = _exec(np.clip(q, lo, hi))
        if term:
            break
    state["noise_on"] = prev
    return term, info


def grasp_at(pos, gh):
    go(np.array([pos[0], pos[1], 0.11]), OPEN, 40)
    go(np.array([pos[0], pos[1], gh]), OPEN, 50, maxstep=0.02)
    hold(CLOSED, 30, noise=False)                       # frozen-arm close, no wobble


def place_and_check(tgt, z0):
    go(np.array([tgt[0], tgt[1], z0 + 0.13]), CLOSED, 45, maxstep=0.025)
    go(np.array([tgt[0], tgt[1], z0 + HALF + 0.006]), CLOSED, 38, maxstep=0.012)
    hold(CLOSED, 6, noise=False)
    for _i in range(66):                                 # slow release ramp (~11 recorded frames)
        _g = CLOSED + (OPEN - CLOSED) * min(1.0, (_i + 1) / 60.0)
        q = np.concatenate([curq()[:5], [_g]]).astype(np.float32)
        state["noise_on"] = False
        _exec(np.clip(q, lo, hi))
    # Hold STILL right after release (recorded); the env terminates AT success, so demos
    # end in the released-on-target success state.
    term, info = hold(OPEN, 60, noise=False)
    if not term:                                        # fallback: retreat then settle unrecorded
        go(np.array([tgt[0], tgt[1], z0 + 0.12]), OPEN, 20)
        rec = state["frames"]
        state["frames"] = None
        _t, info = hold(OPEN, 40, noise=False)
        state["frames"] = rec
    return bool(info.get("success"))


def episode(seed):
    state["obs"], _ = env.reset(seed=seed)
    rng = np.random.default_rng(seed * 7919 + 13)
    state["rng"] = rng
    state["step"] = 0
    state["frames"] = None
    state["noise_on"] = True
    cube = objp()
    if float(np.hypot(cube[0], cube[1])) > 0.27:         # outside the expert's reliable reach
        return None, "far", None
    apply_randomization(rng)
    tgt = tgtp()
    z0 = float(U._initial_obj_z)
    gh = HALF - 0.004
    kind = rng.choice(["std", "retry", "midtask", "release"], p=[0.40, 0.15, 0.15, 0.30])

    if kind in ("midtask", "release"):                   # early phases UNRECORDED
        grasp_at(cube, gh)
        info, _ = go(np.array([cube[0], cube[1], gh + 0.13]), CLOSED, 40, maxstep=0.02)
        if float(info.get("lift_height", 0)) < 0.03:
            return None, "nolift", None
        if kind == "release":                            # also carry UNRECORDED; record from descent
            go(np.array([tgt[0], tgt[1], z0 + 0.13]), CLOSED, 45, maxstep=0.025)
        state["frames"] = []
    else:
        state["frames"] = []
        if kind == "retry":                              # engineered miss -> reopen -> real grasp
            off = rng.uniform(0.022, 0.034) * rng.choice([-1., 1.])
            miss = cube.copy()
            miss[1] += off
            grasp_at(miss, gh)
            info, _ = go(np.array([miss[0], miss[1], gh + 0.10]), CLOSED, 25, maxstep=0.02)
            if float(info.get("lift_height", 0)) > 0.03:  # accidentally grasped anyway
                kind = "std"
            else:
                hold(OPEN, 10, noise=False)
        cube_now = objp()
        if float(np.hypot(cube_now[0], cube_now[1])) > 0.27:
            state["frames"] = None
            return None, "pushed_far", None
        grasp_at(cube_now, gh)
        info, _ = go(np.array([objp()[0], objp()[1], gh + 0.13]), CLOSED, 40, maxstep=0.02)
        if float(info.get("lift_height", 0)) < 0.03:
            state["frames"] = None
            return None, "nolift", None

    succ = place_and_check(tgt, z0)
    frames = state["frames"]
    state["frames"] = None
    if not succ or len(frames) < 8:
        return None, "nosucc", None
    return frames, "ok", kind


names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
features = {
    "observation.images.cam0": {"dtype": "video", "shape": (480, 640, 3), "names": ["height", "width", "channels"]},
    "observation.images.cam1": {"dtype": "video", "shape": (480, 640, 3), "names": ["height", "width", "channels"]},
    "observation.state": {"dtype": "float32", "shape": (6,), "names": names},
    "action": {"dtype": "float32", "shape": (6,), "names": names},
}
ds = LeRobotDataset.create(args.repo_id, fps=33, features=features, root=ROOT,
                           robot_type="so101", use_videos=True)
kept, seed = 0, 0
counts = {"std": 0, "retry": 0, "midtask": 0, "release": 0}
while kept < args.n and seed < args.seed_cap:
    frames, why, kind = episode(seed)
    if frames is None:
        seed += 1
        continue
    for (c0, w, st, ac) in frames:
        ds.add_frame({"observation.images.cam0": c0, "observation.images.cam1": w,
                      "observation.state": st, "action": ac, "task": args.task})
    ds.save_episode()
    counts[kind] += 1
    print(f"seed{seed}: kept ep{kept:03d} [{kind}] ({len(frames)} frames)", flush=True)
    kept += 1
    seed += 1
print(f"DONE kept={kept} attempted={seed} counts={counts} root={ROOT}")
