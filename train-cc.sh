#!/bin/bash
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=2
#SBATCH --mem=25G
module load python/3.8
module load miniconda3
source $HOME/clevr-iep/bin/activate
mkdir $SLURM_TMPDIR/kaggle2015
cp /home/mweiss10/scratch/clevr-txt-v1.tar.xz $SLURM_TMPDIR/clevr-txt
unzip $SLURM_TMPDIR/clevr-txt-v1.tar.xz -d $SLURM_TMPDIR/clevr-txt
mv $SLURM_TMPDIR/kaggle2015/resized_train15 $SLURM_TMPDIR/kaggle2015/train
mv $SLURM_TMPDIR/kaggle2015/trainLabels15.csv $SLURM_TMPDIR/kaggle2015/trainLabels.csv
cd /network/home/luckmarg/fundus-eye-test
python train.py --dataset_dir $SLURM_TMPDIR/kaggle2015 --cfg /network/home/luckmarg/exp-fundus/2015/224_lr01.yml --results_dir /network/home/luckmarg/results_fundus/2015

