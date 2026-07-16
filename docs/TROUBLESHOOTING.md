# Troubleshooting

Common pitfalls, in the order you are likely to hit them. Each one cost us real time;
the fixes are tested.

## 1. The gripper convention is inverted

**Symptom:** grasp rates look plausible but the videos show the jaws opening onto the cube.
**Cause:** on so101-nexus, `-10 deg` is *closed* and `+100 deg` is *open*. The obvious
assumption is backwards, and nothing errors.
**Fix:** run `python collect/measure_gripper.py` and read the gap table before trusting any
number. The mapping used everywhere in this repo: checkpoint channel `0..45` <->
sim `-10..+100 deg` (`g2model` / `model2gdeg` in the scripts).

## 2. `normalize_gripper=false` raises ValueError

**Symptom:** `ValueError` about gripper values outside `[-1, 1]` when training.
**Cause:** that flag expects the dataset gripper channel to already be in `[-1, 1]`.
**Fix:** keep `normalize_gripper=true`. With a binary-relabeled dataset ({0, 45}), the
q01/q99 quantile normalization maps the two values exactly onto -1/+1, which is the clean
two-class target you want.

## 3. Zero-shot needs the exact config strings

**Symptom:** zero-shot runs but behaves randomly.
**Cause:** normalization statistics are selected by string tags; a mismatch is silent.
**Fix:** for `allenai/MolmoAct2-SO100_101`, pass exactly
`norm_tag="so100_so101_molmoact2"`,
`setup_type="single so100/so101 robotic arm in molmoact2"`,
`control_mode="absolute joint pose"`. See `build_policy` in `eval/eval_policy.py`.

## 4. Short eval episodes hide ability

**Symptom:** low grasp/success rates, videos end mid-reach.
**Cause:** the policy is slower than the scripted expert; too short a step budget cuts the
episode off before the place-and-release.
**Fix:** `eval/eval_policy.py` defaults to `--max-pol 240` policy steps. At the ~33 Hz
policy rate (200 Hz sim, 6 substeps per step) that is about 7.3 simulated seconds; the
saved mp4 plays at 50 fps, so it looks about 4x slow (see #13). The same checkpoint scored
lower grasp under a shorter step budget than with the full 240 steps, so give the policy
enough steps to finish before scoring it. (The per-budget grasp figures are from our own eval runs.)

## 5. Headless rendering fails on a fresh GPU box

**Symptom:** `mujoco.FatalError` / EGL errors on a headless GPU box with no display.
**Cause:** missing GL libraries; MuJoCo needs EGL for headless render.
**Fix:**
```
apt-get install -y libglvnd0 libgl1 libglx0 libegl1 libopengl0
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
```
The scripts set the env vars themselves; the apt packages are on you.

## 6. `inference_action_mode` is not saved at train time

**Symptom:** a checkpoint trained with `action_mode=both` produces garbage actions at eval.
**Cause:** the inference-side mode is not persisted with the checkpoint.
**Fix:** set `policy.config.inference_action_mode = "continuous"` after
`from_pretrained` (eval script already does).

## 7. Chunk size and re-query interval must agree

**Symptom:** jittery or overshooting rollouts.
**Cause:** training used 10-step chunks, eval re-queries at a different interval.
**Fix:** train with `chunk_size=10, n_action_steps=10`; eval re-queries every 10 steps
(`--n-action-steps 10`, the default).

## 8. Relabeled datasets must rewrite stats

**Symptom:** a relabeled (binary/delta) dataset trains but the model mis-scales actions.
**Cause:** `meta/stats.json` still holds the source dataset's quantiles; normalization is
computed from them.
**Fix:** use `collect/make_variants.py`, which rewrites min/max/mean/std and all quantiles.
If you relabel any other way, rewrite the stats yourself.

## 9. Eval re-downloads the 21 GB base model

**Symptom:** evaluating a fine-tuned checkpoint pulls `allenai/MolmoAct2-SO100_101` again.
**Cause:** the checkpoint's `config.json` and `policy_preprocessor.json` store
`checkpoint_path` as the hub id.
**Fix:** either accept the one-time download (it caches), or point `checkpoint_path` in
both files at your local cached copy.

## 10. Video decoding errors while training

**Symptom:** dataloader crashes decoding dataset videos.
**Cause:** backend-dependent; torchcodec/ffmpeg combinations vary across images.
**Fix:** `--dataset.video_backend=pyav` (already in `configs/train_v5_bingrip.sh`).

## 11. Determinism across eval runs

**Symptom:** the same checkpoint scores differently run to run.
**Cause:** flow-matching sampling is stochastic by default.
**Fix:** the eval script sets `per_episode_seed=true`, `eval_seed=1000`, and seeds torch
per rollout. Identical seeds then reproduce identical rollouts.

## 12. Some seeds spawn the cube out of reach

**Symptom:** occasional episodes where nothing could have succeeded.
**Cause:** the env can spawn the cube beyond the expert's reliable IK reach.
**Fix:** collector and eval both skip resets with cube radius > 0.27 m. Keep the filter
when you change tasks, or retune it for your object.

## 13. The videos look like slow motion

**Symptom:** saved eval clips play about 4x slower than the sim moved.
**Cause:** the sim runs at 200 Hz and every frame is written, but the mp4 plays at 50 fps.
**Fix:** nothing is wrong; it plays at about 4x slow motion. Multiply by 4 for wall-clock speed,
or subsample frames if you want real-time video.
