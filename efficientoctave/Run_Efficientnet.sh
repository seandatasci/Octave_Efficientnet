# Copyright 2019 seandatasci and Antony Sagayaraj. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
