#!/bin/bash
# make feature_h5 clevr-text-simple
python scripts/extract_features.py --input_image_dir ../clevr-dataset-gen/output/clevr-text-simple/ --output_h5_file data/train_clevr_text_simple_features.h5 --batch_size 10 --multi_dir
python scripts/extract_features.py --input_image_dir ../clevr-dataset-gen/output/clevr-text-simple/ --output_h5_file data/val_clevr_text_simple_features.h5 --batch_size 10 --multi_dir
python scripts/extract_features.py --input_image_dir ../clevr-dataset-gen/output/clevr-text-simple/ --output_h5_file data/test_clevr_text_simple_features.h5 --batch_size 10 --multi_dir

# make feature_h5 clevr-text
python scripts/extract_features.py --input_image_dir ../clevr-dataset-gen/output/train-clevr-text/images/ --output_h5_file data/train_clevr_text_features.h5 --batch_size 10
python scripts/extract_features.py --input_image_dir ../clevr-dataset-gen/output/val-clevr-text/images/ --output_h5_file data/val_clevr_text_features.h5 --batch_size 10
python scripts/extract_features.py --input_image_dir ../clevr-dataset-gen/output/test-clevr-text/images/ --output_h5_file data/test_clevr_text_features.h5 --batch_size 10

# make image h5 clevr-text
python scripts/make_image_h5.py --input_image_dir ../clevr-dataset-gen/output/train-clevr-text/images/ --output_h5_file data/train_clevr_text_images.h5
python scripts/make_image_h5.py --input_image_dir ../clevr-dataset-gen/output/val-clevr-text/images/ --output_h5_file data/val_clevr_text_images.h5
python scripts/make_image_h5.py --input_image_dir ../clevr-dataset-gen/output/test-clevr-text/images/ --output_h5_file data/test_clevr_text_images.h5

# process questions clevr-text-simple
python scripts/preprocess_questions.py   --input_questions_json ../clevr-dataset-gen/output/clevr-text-simple --input_scenes_json ../clevr-dataset-gen/output/clevr-text-simple/   --output_h5_file data/train_questions.h5   --output_vocab_json data/vocab.json --multi_dir --binary_qs_only
python scripts/preprocess_questions.py   --input_questions_json ../clevr-dataset-gen/output/clevr-text-simple --input_scenes_json ../clevr-dataset-gen/output/clevr-text-simple/   --output_h5_file data/val_questions.h5   --input_vocab_json data/vocab.json --multi_dir --binary_qs_only
python scripts/preprocess_questions.py   --input_questions_json ../clevr-dataset-gen/output/clevr-text-simple --input_scenes_json ../clevr-dataset-gen/output/clevr-text-simple/   --output_h5_file data/test_questions.h5   --input_vocab_json data/vocab.json --multi_dir --binary_qs_only

# process questions clevr-text-simple-2
python scripts/preprocess_questions.py   --input_questions_json ../clevr-dataset-gen/output/clevr-text-simple-2 --input_scenes_json ../clevr-dataset-gen/output/clevr-text-simple-2/   --output_h5_file data/train_questions.h5   --output_vocab_json data/vocab.json --multi_dir --binary_qs_only
python scripts/preprocess_questions.py   --input_questions_json ../clevr-dataset-gen/output/clevr-text-simple-2 --input_scenes_json ../clevr-dataset-gen/output/clevr-text-simple-2/   --output_h5_file data/val_questions.h5   --input_vocab_json data/vocab.json --multi_dir --binary_qs_only
python scripts/preprocess_questions.py   --input_questions_json ../clevr-dataset-gen/output/clevr-text-simple-2 --input_scenes_json ../clevr-dataset-gen/output/clevr-text-simple-2/   --output_h5_file data/test_questions.h5   --input_vocab_json data/vocab.json --multi_dir --binary_qs_only

# process questions clevr-text
python scripts/preprocess_questions.py   --input_questions_json ../clevr-dataset-gen/output/train-clevr-text/CLEVR_questions.json --input_scenes_json ../clevr-dataset-gen/output/train-clevr-text/CLEVR_scenes.json   --output_h5_file data/train_questions.h5   --output_vocab_json data/vocab.json
python scripts/preprocess_questions.py   --input_questions_json ../clevr-dataset-gen/output/val-clevr-text/CLEVR_questions.json --input_scenes_json ../clevr-dataset-gen/output/val-clevr-text/CLEVR_scenes.json   --output_h5_file data/val_questions.h5   --input_vocab_json data/vocab.json
python scripts/preprocess_questions.py   --input_questions_json ../clevr-dataset-gen/output/test-clevr-text/CLEVR_questions.json --input_scenes_json ../clevr-dataset-gen/output/test-clevr-text/CLEVR_scenes.json   --output_h5_file data/test_questions.h5   --input_vocab_json data/vocab.json
