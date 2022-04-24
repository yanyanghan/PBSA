#!/bin/bash

DATA_DIR="PATH/TO/IMAGENET/val"

# run baseline attack
python test.py \
  --test_dir "$DATA_DIR" \
  --src_model deit_tiny_patch16_224 \
  --tar_model tnt_s_patch16_224  \
  --attack_type mifgsm \
  --eps 16 \
  --index "last" \
  --batch_size 128

# run self-ensemble attack
python test.py \
  --test_dir "$DATA_DIR" \
  --src_model deit_tiny_patch16_224 \
  --tar_model tnt_s_patch16_224  \
  --attack_type mifgsm \
  --eps 16 \
  --index "all" \
  --batch_size 128

# run self-ensemble attack with token-refinement
python test.py \
  --test_dir "$DATA_DIR" \
  --src_model tiny_patch16_224_hierarchical \
  --tar_model tnt_s_patch16_224  \
  --attack_type mifgsm \
  --eps 16 \
  --index "all" \
  --batch_size 128

# run pma
python test.py --src_model deit_tiny_patch16_224 --tar_model tnt_s_patch16_224  --attack_type pgd --eps 16 --index "pma" --batch_size 128 --test_dir "data/train" --m 5 --k 5
default=["T2t_vit_24", "tnt_s_patch16_224", "vit_small_patch16_224", "T2t_vit_7", "resnet152", "senet154"]
default=["resnet152", "resnet50", "vgg19", "densenet201", "senet154"]
CUDA_VISIBLE_DEVICES=0,1 python test.py --src_model deit_tiny_patch16_224 --attack_type pifgsm --eps 16 --index "last" --batch_size 128 --test_dir "data/train" --m 5 --k 5

CUDA_VISIBLE_DEVICES=5,6 python test.py  --src_model tiny_patch16_224_hierarchical --attack_type fgsm --eps 16 --index "all" --batch_size 128 --test_dir "data/train"