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
#SBATCH -J Download_Imagenet

# Specify an output file
# %j is a special variable that is replaced by the JobID when
# job starts
#SBATCH -o Download_Imagenet-%j.out
#SBATCH -e Download_Imagenet-%j.out

#----- End of slurm commands ----

# wget http://www.image-net.org/challenges/LSVRC/2012/nnoupb/ILSVRC2012_img_train.tar

# wget http://www.image-net.org/challenges/LSVRC/2012/nnoupb/ILSVRC2012_img_val.tar

module unload python/2.7.12
module load cuda/10.0.130
module load cudnn/7.4

conda activate efficientoctave

python imagenet_to_gcs.py \
  --local_scratch_dir="./imagenet" \
  --raw_data_dir="./imagenet/raw_data" \
  --imagenet_username=asagayar \
  --imagenet_access_key=3170eff67b6db60088681ccd516652298eb7cceb \

echo "Finished Preparing Data"
echo $(date)
