# molmoact2-so101-sim

Run, fine-tune, and evaluate [MolmoAct2](https://allenai.org/blog/molmoact2) on a simulated
SO-101 arm.

MolmoAct2 is Ai2's open 5B vision-language-action model, shipped in
[LeRobot](https://github.com/huggingface/lerobot). The
[so101-nexus](https://github.com/johnsutor/so101-nexus) MuJoCo sim gives you an SO-101
without hardware. Out of the box the two don't talk to each other: the checkpoint speaks a
real-arm joint convention, the sim speaks its own, and there is no collector, training
config, or eval loop connecting them. This repo is that missing glue, plus everything we
learned making it work: from 0% zero-shot to 93% grasp in five days on one 24 GB GPU,
with no human demonstrations.

Full story with videos: **[project page](https://ataghof.github.io/molmoact2-so101-sim/)**.

| model | recipe | grasp | success |
|---|---|---|---|
| zero-shot | no fine-tuning | 0% | 0% |
| v2 | 300 scripted demos + wrist camera | 33% | 0% |
| v4 | + binary gripper + delta actions | 70-77% | 15% |
| v5 | all weights instead of LoRA | 40% | ~0% |
| v6 | binary gripper, no delta | 93% | 30% |

Grasp = cube held in the gripper at some point. Success = cube placed on the target with the
arm at rest, on cube positions never seen in training. On the looser metric
(released within 6 cm of the target), v6 finishes half its runs. The downloadable model is
v6.

## What's in the box

| path | what it is |
|---|---|
| `src/molmoact2_so101_sim/` | the bridge: joint calibration transform + guards (`adapter.py`, `calibration.py`), sim scene/camera matching (`realism.py`) |
| `collect/` | scripted-expert demo collector (no teleop), gripper-convention measurement, action relabeling (`make_variants.py`) |
| `eval/` | closed-loop rollout + scoring (grasp / success / loose), per-seed videos |
| `configs/` | the known-good training command with the measured ablations noted |
| `evidence/` | tensor-level input diff vs LIBERO: the input pipelines matched, so the photorealism pass was not what moved success (`evidence/`) |
| `docs/` | project page + [TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md), 13 pitfalls with fixes |
| `tests/` | CPU-only unit tests for the calibration bridge |

Weights and dataset on Hugging Face:
[v6 champion](https://huggingface.co/ataghof/molmoact2-so101nexus-lora-champion) ·
[dataset](https://huggingface.co/datasets/ataghof/so101nexus-cube500-binary).
The quickstart below trains v6, the downloadable recipe.

## Quickstart

Collection runs on CPU (a laptop is fine). Training and eval want a 24 GB GPU.

```bash
git clone https://github.com/ataghof/molmoact2-so101-sim && cd molmoact2-so101-sim
pip install -e '.[sim,molmoact2]'

# 0. measure the gripper convention (do this once, see why in TROUBLESHOOTING #1)
python collect/measure_gripper.py

# 1. collect 500 scripted demos (a few hours, CPU)
python collect/collect_demos.py --n 500 --root data/cube500 --repo-id local/cube500

# 2. relabel the gripper to binary open/closed (the single biggest win we measured)
python collect/make_variants.py --src data/cube500 --dst data/cube500_bin --binary

# 3. fine-tune (~4 h on a 24 GB GPU) and evaluate with videos
bash configs/train_v5_bingrip.sh
python eval/eval_policy.py --policy outputs/v5_bingrip/checkpoints/004000/pretrained_model \
    --dataset-root data/cube500_bin --dataset-repo-id local/cube500_bin \
    --seeds 5000-5099 --tag mine --video
```

Zero-shot baseline, no training:

```bash
python eval/eval_policy.py --policy zeroshot --dataset-root data/cube500 \
    --dataset-repo-id local/cube500 --seeds 5000-5099 --tag zeroshot
```

(The config and output paths keep the internal `v5_bingrip` name; it is the v6 recipe.)

### The delta ablation (v4)

v6 uses absolute joint actions. v4 adds delta actions on top of the same binary gripper, and
they were measured to *hurt* grasp on this task (93% to 77%), which is why v6 drops them. To
reproduce v4, relabel with `--binary --delta` and train with a delta control mode:

```bash
python collect/make_variants.py --src data/cube500 --dst data/cube500_bindelta --binary --delta
# then train with --policy.control_mode='delta joint pose' on data/cube500_bindelta
# (see the "delta actions" ablation note in configs/train_v5_bingrip.sh)
```

## Changing the task

The loop is not cube-specific. To point it at your own task:

1. Edit the object and prompt in `collect/collect_demos.py` (`CubeObject`, `--task`).
2. Rewrite the scripted expert (`grasp_at` / `place_and_check`) for your motion. If you can
   script it once, the collector turns it into a dataset.
3. The dataset choices that helped on the cube task (DART noise, retry episodes,
   success-state endings, slow release, binary gripper) are a good starting point for yours.
4. Train and eval with the same commands.

## What actually mattered

The controlled changes are documented on the project page. The short version, for the cube
task only:

- The photorealism pass did not move our success rate. The changes that helped were all in
  the training data, not the pixels (`evidence/`).
- The binary gripper relabel gave the single largest gain we measured.
- Delta actions looked fine inside the full stack and cost about 16 points of grasp when
  isolated (93% to 77%).
- LoRA beat full fine-tuning at this data scale.
- Longer eval episodes raised grasp with the same weights (reported 53% to 77%).

## Related work

The pieces existed separately. What this repo adds is the sim adapter and the fine-tuning
recipe that connect them.

- [irenegracekp/molmoact2-so101](https://github.com/irenegracekp/molmoact2-so101) runs
  MolmoAct2 zero-shot on a **real** SO-101. We used it as the reference for the
  observation/action contract and joint calibration that our bridge implements in sim.
- [EE5108-DigitalTwins/lerobot_mujoco_sim](https://github.com/EE5108-DigitalTwins/lerobot_mujoco_sim)
  does closed-loop SO-101 pick-and-place in MuJoCo with an **ACT** policy: the nearest sim
  scaffold, with a different (non-VLA) policy. We built our own harness rather than forking it.
- [allenai/molmoact2 `sim_eval/`](https://github.com/allenai/molmoact2/tree/main/sim_eval)
  and the [MolmoAct2-LIBERO checkpoint](https://huggingface.co/allenai/MolmoAct2-LIBERO-LeRobot)
  are the working sim-eval template (on LIBERO/Franka) our eval loop follows.

What we did not find, and built: MolmoAct2 driven closed-loop inside an SO-101 *simulator*,
with fine-tuned success numbers.

## Requirements

- Python 3.12
- Collection: `so101-nexus`, MuJoCo 3.10+ (tested with 3.10), lerobot 0.6+ (CPU is fine)
- Training/eval: CUDA GPU, 24 GB for LoRA at batch 8. The base checkpoint download is ~21 GB.
- Headless boxes need GL libraries, see [TROUBLESHOOTING #5](docs/TROUBLESHOOTING.md).

## Acknowledgements

[MolmoAct2](https://allenai.org/blog/molmoact2) by Ai2 ·
[LeRobot](https://github.com/huggingface/lerobot) by Hugging Face ·
[so101-nexus](https://github.com/johnsutor/so101-nexus) MuJoCo SO-101 sim by John Sutor ·
[SO-101 arm](https://github.com/TheRobotStudio/SO-ARM100) by TheRobotStudio.

Apache-2.0.
