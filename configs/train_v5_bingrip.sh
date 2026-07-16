#!/usr/bin/env bash
# Train the best recipe from this repo's experiments: "v5_bingrip".
# NOTE: "v5_bingrip" is the internal name for the model called v6 on the project page and
# the downloadable champion on Hugging Face. Same recipe, kept as-is so paths don't break.
# LoRA on the VLM + full training of the action expert, binary-gripper dataset,
# 10-step action chunks, absolute joint targets. Measured: 93% grasp / 30% success
# on held-out cube positions (30 runs). Fits a 24 GB GPU; ~4 h at these settings.
#
# Prerequisites:
#   pip install -e '.[sim,molmoact2]'
#   python collect/collect_demos.py --n 500 --root data/cube500 --repo-id local/cube500
#   python collect/make_variants.py --src data/cube500 --dst data/cube500_bin --binary
#
# Ablations measured on this task, for the curious:
#   delta actions   change --dataset to a --binary --delta variant and
#                   --policy.control_mode='delta joint pose'  -> WORSE (70-77% grasp)
#   full fine-tune  drop --policy.enable_lora_vlm             -> WORSE (40% grasp, no
#                   real successes; needs ~96 GB)
set -euo pipefail

REPO_ID=${REPO_ID:-local/cube500_bin}
DATA_ROOT=${DATA_ROOT:-data/cube500_bin}
OUT_DIR=${OUT_DIR:-outputs/v5_bingrip}
STEPS=${STEPS:-4000}

lerobot-train \
  --dataset.repo_id="$REPO_ID" --dataset.root="$DATA_ROOT" \
  --dataset.video_backend=pyav --dataset.image_transforms.enable=true \
  --policy.type=molmoact2 --policy.checkpoint_path=allenai/MolmoAct2-SO100_101 \
  --policy.device=cuda \
  --policy.action_mode=both --policy.enable_lora_vlm=true \
  --policy.chunk_size=10 --policy.n_action_steps=10 \
  --policy.setup_type='single so100/so101 robotic arm in molmoact2' \
  --policy.control_mode='absolute joint pose' \
  --policy.image_keys='["observation.images.cam0","observation.images.cam1"]' \
  --policy.model_dtype=bfloat16 --policy.num_flow_timesteps=8 \
  --policy.gradient_checkpointing=true --policy.normalize_gripper=true \
  --policy.optimizer_action_expert_lr=5e-5 \
  --policy.push_to_hub=false --wandb.enable=false \
  --job_name=v5_bingrip --output_dir="$OUT_DIR" \
  --steps="$STEPS" --batch_size=8 --num_workers=4 --log_freq=100 \
  --save_checkpoint=true --save_freq="$STEPS"

echo "checkpoint: $OUT_DIR/checkpoints/$(printf '%06d' "$STEPS")/pretrained_model"
echo "next: python eval/eval_policy.py --policy <that path> --dataset-root $DATA_ROOT --dataset-repo-id $REPO_ID --seeds 5000-5099 --tag v5_bingrip --video"
