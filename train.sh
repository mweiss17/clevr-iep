#!/bin/bash

python scripts/train_model.py \
  --model_type PG \
  --num_train_samples 18000 \
  --num_iterations 3000 \
  --checkpoint_every 1000 \
  --train_ocr_token_json ../clevr-dataset-gen/output/clevr-text-simple/CLEVR_ocr.json   \
  --val_ocr_token_json ../clevr-dataset-gen/output/clevr-text-simple/CLEVR_ocr.json   \
  --checkpoint_path data/program_generator.pt


python scripts/train_model.py \
  --model_type EE \
  --program_generator_start_from data/program_generator.py \
  --num_iterations 100000 \
  --train_ocr_token_json ../clevr-dataset-gen/output/clevr-text-simple/CLEVR_ocr.json   \
  --val_ocr_token_json ../clevr-dataset-gen/output/clevr-text-simple/CLEVR_ocr.json   \
  --train_features_h5 data/train_clevr_text_simple_features.h5 \
  --val_features_h5 data/val_clevr_text_simple_features.h5 \
  --checkpoint_path data/execution_engine.pt


python scripts/train_model.py \
  --model_type PG+EE+GQNT \
  --num_train_samples 18000 \
  --num_iterations 20000 \
  --checkpoint_every 1000 \
  --checkpoint_path data/gqnt.pt \
  --train_ocr_token_json ../clevr-dataset-gen/output/train-clevr-text/CLEVR_ocr.json \
  --val_ocr_token_json ../clevr-dataset-gen/output/val-clevr-text/CLEVR_ocr.json \
  --train_features_h5 data/train_clevr_text_features.h5 \
  --val_features_h5 data/val_clevr_text_features.h5 \
#  --train_images_h5 data/train_clevr_text_images.h5 \
#  --val_images_h5 data/val_clevr_text_images.h5 \

# salloc --time=1:0:0 --acount=rpp-bengioy --gres=gpu:2 --cpus-per-task=2 --mem=20G

CUDA_VISIBLE_DEVICES=1 python scripts/train_model.py  \
  --model_type PG+EE+GQNT   \
  --num_train_samples 18000   \
  --num_iterations 20000   \
  --checkpoint_every 1000   \
  --checkpoint_path data/gqnt.pt  \
  --train_ocr_token_json ../clevr-dataset-gen/output/clevr-text-simple/CLEVR_ocr.json   \
  --val_ocr_token_json ../clevr-dataset-gen/output/clevr-text-simple/CLEVR_ocr.json   \
  --train_features_h5 data/train_clevr_text_simple_features.h5 \
  --val_features_h5 data/val_clevr_text_simple_features.h5