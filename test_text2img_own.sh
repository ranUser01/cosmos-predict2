#!/bin/sh 
### General options 
### -- specify queue -- 
#BSUB -q gpul40s ### gpua40 ### gpuv100 ###gpua100
### need at least 48 gb``
###BSUB -R "select[gpu48gb]"
### -- set the job Name -- 
#BSUB -J DDPM
### -- ask for number of cores (default: 1) -- 
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
### -- specify that the cores must be on the same host -- 
#BSUB -R "span[hosts=1]"
### -- specify that we need 4GB of memory per core/slot -- 
#BSUB -R "rusage[mem=8GB]"
### -- specify that we want the job to get killed if it exceeds 5 GB per core/slot -- 
#BSUB -M 9GB
### -- set walltime limit: hh:mm -- 
#BSUB -W 23:00 
### -- Specify the output and error file. %J is the job-id -- 
#BSUB -o Test_World%J.out
#BSUB -e Test_World%J.err

# module load pandas/2.1.3-python-3.10.13

module load cuda/12.4
nvcc --version
echo $CUDA_HOME
export CUDA_HOME=/apps/dcc/cuda/12.4   # replace with actual module path if needed
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

export HF_HOME="/work3/s243891/huggingface/datasets"

# Make conda available in this shell
source /work3/s243891/miniforge3/etc/profile.d/conda.sh

# Activate your environment
conda activate CosmosPredictNvidia

./scripts/hf_text2image.py /work3/s243891/output/hf_text2image \
    --prompt prompts/own.txt
    -v
