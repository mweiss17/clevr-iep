#!/bin/bash
module load miniconda/3
source activate clevr-iep
python -u scripts/train_model.py \
  --model_type PG \
  --num_train_samples 500000 \
  --num_iterations 200000 \
  --checkpoint_every 10000 \
  --checkpoint_path data/program_generator_500k.pt \
  --train_features_h5 /network/tmp1/weissmar/train_features.h5
