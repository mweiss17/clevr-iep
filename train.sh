#!/bin/bash

#python scripts/train_model.py \
#  --model_type PG \
#  --num_train_samples 596 \
#  --num_iterations 3000 \
#  --checkpoint_every 1000 \
#  --checkpoint_path data/program_generator.pt


python scripts/train_model.py \
  --model_type PG+EE+GQNT \
  --num_train_samples 18000 \
  --num_iterations 20000 \
  --checkpoint_every 1000 \
  --checkpoint_path data/gqnt.pt \
  --train_images_h5 data/train_clevr_text_images.h5 \
  --val_images_h5 data/val_clevr_text_images.h5 \
  --train_ocr_token_json ../clevr-dataset-gen/output/train-clevr-text/CLEVR_ocr.json \
  --val_ocr_token_json ../clevr-dataset-gen/output/val-clevr-text/CLEVR_ocr.json \
