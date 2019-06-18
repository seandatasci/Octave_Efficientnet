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
