#!/bin/bash
module load miniconda/3
source activate clevr-iep
python -u scripts/train_model.py   --model_type CNN+LSTM+SA   --classifier_fc_dims 1024   --num_iterations 400000   --checkpoint_path data/cnn_lstm_sa_mw2.pt --train_features_h5 /network/tmp1/weissmar/train_features.h5
