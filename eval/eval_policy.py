"""Roll out a MolmoAct2 policy (fine-tuned checkpoint or zero-shot) in so101-nexus and score it.

Same sim setup as demo collection (realism tuning, front_low cam0 + wrist cam1, 3 cm cube,
friction fix). The policy runs at ~33 Hz: each action is held for 6 sim substeps to match
the 200 Hz -> 33 Hz training cadence. Actions arrive in the checkpoint convention and are
inverted back to the sim frame.

Reported per run and in summary:
    grasp    the gripper held the cube at some point
    success  cube placed on the target and the arm at rest (the sim's own criterion)
    loose    success OR cube released within 6 cm of the target

Episodes default to 240 policy steps. Short eval clocks hide ability: on this task the
same checkpoint measured 53% grasp at ~10 s episodes and 77% with room to finish.

Usage:
    python eval/eval_policy.py --policy zeroshot --dataset-root data/cube500 --tag zeroshot
    python eval/eval_policy.py --policy outputs/ckpt/pretrained_model --seeds 5000-5099 --tag mine --video
"""
import argparse
import json
import os
import pathlib
import sys

if sys.platform == "linux":                    # EGL is Linux-only; macOS picks its native GL
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image

from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.policies import make_pre_post_processors
from lerobot.policies.factory import make_policy, make_policy_config
from lerobot.policies.molmoact2.modeling_molmoact2 import MolmoAct2Policy
from so101_nexus import OverheadCamera, WristCamera
from so101_nexus.mujoco.pick_and_place import PickAndPlaceEnv, PickAndPlaceConfig
from so101_nexus.objects import CubeObject

from molmoact2_so101_sim import realism

HALF = 0.015
SUB = 6
JS = np.array([1., -1., 1., 1., 1., 1.])
JO = np.array([0., 90., 90., 0., 0., 0.])
TASK = "Pick up the red cube and place it on the blue circle."


def g2model(gdeg):
    """Sim gripper angle (deg) -> checkpoint gripper channel (0..45)."""
    return float(np.clip((gdeg + 10.) / 110. * 45., 0, 45))


def model2gdeg(gm):
    """Checkpoint gripper channel (0..45) -> sim gripper angle (deg)."""
    return gm / 45. * 110. - 10.


def build_policy(spec, meta, control_mode=None, n_action_steps=10):
    """`spec` is 'zeroshot' or a path to a trained checkpoint's pretrained_model dir."""
    if spec == "zeroshot":
        cfg = make_policy_config(
            "molmoact2", checkpoint_path="allenai/MolmoAct2-SO100_101",
            norm_tag="so100_so101_molmoact2", model_dtype="bfloat16",
            chunk_size=30, n_action_steps=30,
            setup_type="single so100/so101 robotic arm in molmoact2",
            control_mode="absolute joint pose",
            image_keys=["observation.images.cam0", "observation.images.cam1"],
            normalize_gripper=True, inference_action_mode="continuous", device="cuda")
        pol = make_policy(cfg, ds_meta=meta)
        pre, post = make_pre_post_processors(policy_cfg=cfg, dataset_stats=meta.stats)
    else:
        pol = MolmoAct2Policy.from_pretrained(spec)
        pol.config.inference_action_mode = "continuous"   # not saved at train time
        if control_mode:
            pol.config.control_mode = control_mode
        pol.to("cuda")
        pre, post = make_pre_post_processors(policy_cfg=pol.config, pretrained_path=spec)
    pol.config.per_episode_seed = True      # deterministic flow-matching sampling
    pol.config.n_action_steps = n_action_steps  # closed-loop re-query interval
    pol.config.eval_seed = 1000
    pol.eval()
    return pol, pre, post


realism.install()
cfg = PickAndPlaceConfig(
    obs_mode="visual",
    observations=list(PickAndPlaceConfig().observations or []) + [OverheadCamera(), WristCamera()],
    objects=[CubeObject(half_size=HALF, color="red")])
env = PickAndPlaceEnv(config=cfg, render_mode="rgb_array")
U = env.unwrapped
realism.tune_arm(U.model)
realism.set_camera(env, "front_low")
model, data = U.model, U.data
lo, hi = env.action_space.low, env.action_space.high
# Same grip-friendly contact params as collection.
for gid in set(U._gripper_geom_ids) | set(U._jaw_geom_ids) | {U._obj_geom_id}:
    f = model.geom_friction[gid].copy()
    model.geom_friction[gid] = [max(f[0], 1.2), max(f[1] * 10, 0.05), max(f[2] * 10, 0.01)]


def objp():
    return U._get_object_pose()[:3].copy()


def obs_to_batch(obs):
    c0 = np.ascontiguousarray(obs["overhead_camera"]).astype(np.uint8)
    w = np.ascontiguousarray(obs["wrist_camera"]).astype(np.uint8)
    c1 = w if w.shape[:2] == (480, 640) else np.asarray(Image.fromarray(w).resize((640, 480))).astype(np.uint8)

    def img(x):
        return torch.from_numpy(x).permute(2, 0, 1).float().div(255.).unsqueeze(0).to("cuda")

    sd = np.rad2deg(obs["state"])
    st = (JS * sd + JO).astype(np.float32)
    st[5] = g2model(sd[5])
    batch = {"observation.images.cam0": img(c0), "observation.images.cam1": img(c1),
             "observation.state": torch.from_numpy(st).unsqueeze(0).to("cuda"), "task": [TASK]}
    return batch, c0


def act_to_sim(a):
    arm = np.deg2rad((a[:5] - JO[:5]) / JS[:5])
    grip = np.deg2rad(model2gdeg(float(a[5])))
    return np.clip(np.concatenate([arm, [grip]]).astype(np.float32), lo, hi)


def rollout(pol, pre, post, seed, frames=None, max_pol=240, delta=False):
    torch.manual_seed(1000)
    torch.cuda.manual_seed_all(1000)
    obs, _ = env.reset(seed=seed)
    pol.reset()
    rollout._reset_cmd = True
    if hasattr(pol, "_rollout_index_for_task"):      # per-episode independent + reproducible
        pol._rollout_index_for_task = -1
        pol._rollout_task_key = None
    if float(np.hypot(*objp()[:2])) > 0.27:          # cube spawned out of reach: skip seed
        return None
    grasped = False
    lifted = 0.0
    info = {}
    for _t in range(max_pol):
        batch, _c0 = obs_to_batch(obs)
        with torch.no_grad():
            if pre is not None:
                proc = pre(dict(batch))
                a_raw = pol.select_action(proc)
                a = post(a_raw)[0].detach().cpu().numpy()
            else:
                a = pol.select_action(batch)[0].detach().cpu().numpy()
        if delta:
            # Integrate delta-arm actions into an absolute command (gripper stays absolute).
            if not hasattr(rollout, "_cmd") or rollout._reset_cmd:
                sd = np.rad2deg(obs["state"])
                rollout._cmd = (JS * sd + JO).astype(np.float32)
                rollout._cmd[5] = g2model(sd[5])
                rollout._reset_cmd = False
            rollout._cmd[:5] = rollout._cmd[:5] + a[:5]
            rollout._cmd[5] = a[5]
            a = rollout._cmd.copy()
        q = act_to_sim(a)
        term = False
        for _ in range(SUB):
            obs, _r, term, _t2, info = env.step(q)
            if frames is not None:
                frames.append(np.ascontiguousarray(obs["overhead_camera"]).astype(np.uint8))
            if term:
                break
        grasped |= bool(info.get("is_grasped"))
        lifted = max(lifted, float(info.get("lift_height", 0)))
        if term:
            break
    return dict(grasp=grasped, lift=lifted, succ=bool(info.get("success")),
                otd=float(info.get("obj_to_target_dist", 9)),
                end_grasped=bool(info.get("is_grasped")))


def parse_seeds(s):
    if "-" in s:
        a, b = s.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in s.split(",")]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", required=True, help="'zeroshot' or a checkpoint pretrained_model dir")
    ap.add_argument("--dataset-root", default="data/cube500",
                    help="a collected dataset dir; zero-shot needs its stats for normalization")
    ap.add_argument("--dataset-repo-id", default="local/cube500")
    ap.add_argument("--seeds", default="5000-5099",
                    help="range 'a-b' or comma list; out-of-reach spawns are skipped (default scores 30)")
    ap.add_argument("--tag", required=True, help="label for output files")
    ap.add_argument("--out", default="eval_out", help="output directory")
    ap.add_argument("--video", action="store_true", help="save mp4 for seeds in --video-seeds")
    ap.add_argument("--video-seeds", default="", help="subset of seeds to record")
    ap.add_argument("--max-pol", type=int, default=240,
                    help="policy steps per episode; short clocks hide ability")
    ap.add_argument("--delta", action="store_true", help="integrate delta-arm actions")
    ap.add_argument("--control-mode", default=None,
                    help="override checkpoint control_mode, e.g. 'delta joint pose'")
    ap.add_argument("--n-action-steps", type=int, default=10, help="closed-loop re-query interval")
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = LeRobotDatasetMetadata(args.dataset_repo_id, root=args.dataset_root)
    print(f"loading policy: {args.policy}")
    pol, pre, post = build_policy(args.policy, meta, control_mode=args.control_mode,
                                  n_action_steps=args.n_action_steps)
    vseeds = set(parse_seeds(args.video_seeds)) if args.video_seeds else set()
    res = []
    for s in parse_seeds(args.seeds):
        frames = [] if (args.video and (not vseeds or s in vseeds)) else None
        r = rollout(pol, pre, post, s, frames=frames, max_pol=args.max_pol, delta=args.delta)
        if r is None:
            continue
        res.append(r)
        print(f"  seed{s}: grasp={r['grasp']} lift={r['lift']:.3f} succ={r['succ']} "
              f"otd={r['otd']:.3f} end_grasped={r['end_grasped']}", flush=True)
        if frames:
            imageio.mimwrite(out / f"{args.tag}_seed{s}.mp4", frames, fps=50, codec="libx264",
                             quality=8, macro_block_size=None, pixelformat="yuv420p")
    n = len(res)
    released = [x for x in res if x["otd"] < 0.06 and not x["end_grasped"]]
    loose = [x for x in res if x["succ"] or (x["otd"] < 0.06 and not x["end_grasped"])]
    summ = dict(tag=args.tag, n=n,
                grasp=sum(x["grasp"] for x in res),
                lift=sum(x["lift"] > 0.03 for x in res),
                succ=sum(x["succ"] for x in res),
                released_6cm=len(released),
                loose=len(loose))
    print(f"EVAL[{args.tag}] n={n} grasp={summ['grasp']}/{n} lift={summ['lift']}/{n} "
          f"succ={summ['succ']}/{n} loose={summ['loose']}/{n}")
    (out / f"summary_{args.tag}.json").write_text(json.dumps(dict(summary=summ, per=res), indent=1))


if __name__ == "__main__":
    main()
