#!/bin/bash
python scripts/extract_features.py --input_image_dir ../clevr-dataset-gen/output/images/ --output_h5_file data/train_features.h5 --batch_size 10
python scripts/preprocess_questions.py   --input_questions_json ../clevr-dataset-gen/output/CLEVR_questions.json --input_scenes_json ../clevr-dataset-gen/output/CLEVR_scenes.json   --output_h5_file data/train_questions.h5   --output_vocab_json data/vocab.json
