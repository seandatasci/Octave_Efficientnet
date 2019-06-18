#!/bin/bash

#
# Once the job starts you will see a file MySerialJob-****.out
# The **** will be the slurm JobID

# --- Start of slurm commands -----------

#SBATCH -p gpu --gres=gpu:1
#SBATCH -n 8
#SBATCH --time=48:00:00
#SBATCH --mem=128G


# Specify a job name:
#SBATCH -J Run_Efficientnet

# Specify an output file
# %j is a special variable that is replaced by the JobID when
# job starts
#SBATCH -o Run_Efficientnet-%j.out
#SBATCH -e Run_Efficientnet-%j.out

#----- End of slurm commands ----

module unload python/2.7.12
module load cuda/10.0.130
module load cudnn/7.4

conda activate efficientoctave
python main.py \
    --data_dir imagenet/train \
    --model_dir models \
    --model_name efficientnet-b0 \
    --mode train_and_eval \
    --train_steps 100 \
    --num_parallel_calls 8


python main.py \
    --data_dir tiny-imagenet/all \
    --model_dir models \
    --model_name efficientnet-b0 \
    --mode train_and_eval \
    --train_steps 100 \
    --train_batch_size 1 \
    --eval_batch_size 1 \
    --num_train_images 100000 \
    --num_eval_images 10000 \
    --num_label_classes 200 \
    --steps_per_eval 2 \
    --num_parallel_calls 8
