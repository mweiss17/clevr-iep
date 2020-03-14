#!/bin/bash
module load miniconda/3
source activate clevr-iep
python -u scripts/train_model.py \
  --model_type EE \
  --program_generator_start_from data/program_generator.py \
  --num_iterations 100000 \
  --checkpoint_path data/execution_engine.pt
