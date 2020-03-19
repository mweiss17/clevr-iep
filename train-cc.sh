#!/bin/bash
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=25G
#SBATCH --account=rpp-bengioy
#SBATCH --time=24:00:00

module load python/3.8
source $HOME/clevr-iep/venv/bin/activate
mkdir $SLURM_TMPDIR/clevr-iep
cp /home/mweiss10/scratch/clevr-text-simple.zip $SLURM_TMPDIR/clevr-iep/clevr-text-simple.zip
unzip $SLURM_TMPDIR/clevr-iep/clevr-text-simple.zip -d $SLURM_TMPDIR/data
cd /home/mweiss10/clevr-iep
python scripts/train_model.py \
  --model_type EE \
  --program_generator_start_from $SLURM_TMPDIR/data/program_generator.py \
  --num_iterations 1000000 \
  --train_ocr_token_json $SLURM_TMPDIR/data/CLEVR_ocr.json   \
  --val_ocr_token_json $SLURM_TMPDIR/data/CLEVR_ocr.json   \
  --train_features_h5 $SLURM_TMPDIR/data/train_clevr_text_simple_features.h5 \
  --val_features_h5 $SLURM_TMPDIR/data/val_clevr_text_simple_features.h5 \
  --checkpoint_path $SLURM_TMPDIR/data/execution_engine.pt \
  --vocab_json $SLURM_TMPDIR/data/vocab.json \
  --train_question_h5 $SLURM_TMPDIR/data/train_questions.h5 \
  --val_question_h5 $SLURM_TMPDIR/data/val_questions.h5 \
  --multi_gpu
