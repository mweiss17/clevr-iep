#!/bin/bash
# process features
#python scripts/extract_features.py --input_image_dir ../clevr-dataset-gen/output/images/ --output_h5_file data/train_features.h5 --batch_size 10
python scripts/make_image_h5.py --input_image_dir ../clevr-dataset-gen/output/train-clevr-text/images/ --output_h5_file data/train_clevr_text_images.h5
python scripts/make_image_h5.py --input_image_dir ../clevr-dataset-gen/output/val-clevr-text/images/ --output_h5_file data/val_clevr_text_images.h5
python scripts/make_image_h5.py --input_image_dir ../clevr-dataset-gen/output/test-clevr-text/images/ --output_h5_file data/test_clevr_text_images.h5

# process questions
python scripts/preprocess_questions.py   --input_questions_json ../clevr-dataset-gen/output/train-clevr-text/CLEVR_questions.json --input_scenes_json ../clevr-dataset-gen/output/train-clevr-text/CLEVR_scenes.json   --output_h5_file data/train_questions.h5   --output_vocab_json data/vocab.json
python scripts/preprocess_questions.py   --input_questions_json ../clevr-dataset-gen/output/val-clevr-text/CLEVR_questions.json --input_scenes_json ../clevr-dataset-gen/output/val-clevr-text/CLEVR_scenes.json   --output_h5_file data/val_questions.h5   --input_vocab_json data/vocab.json
python scripts/preprocess_questions.py   --input_questions_json ../clevr-dataset-gen/output/test-clevr-text/CLEVR_questions.json --input_scenes_json ../clevr-dataset-gen/output/test-clevr-text/CLEVR_scenes.json   --output_h5_file data/test_questions.h5   --input_vocab_json data/vocab.json
