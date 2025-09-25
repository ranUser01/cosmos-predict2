#!/bin/sh 
### General options 
### -- specify queue -- 
#BSUB -q gpul40s ### gpua40 ### gpuv100 ###gpua100
### need at least 48 gb for 14B model
###BSUB -R "select[gpu48gb]"
### -- set the job Name -- 
#BSUB -J BigSpace
### -- ask for number of cores (default: 1) -- 
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
### -- specify that the cores must be on the same host -- 
#BSUB -R "span[hosts=1]"
### -- specify that we need memory per core/slot (adjust based on model size) -- 
#BSUB -R "rusage[mem=4GB]"
### -- specify that we want the job to get killed if it exceeds memory limit -- 
#BSUB -M 5GB
### -- set walltime limit: hh:mm (increased for 14B model) -- 
#BSUB -W 23:00 
### -- Specify the output and error file. %J is the job-id -- 
#BSUB -o BigSpace_%J.out
#BSUB -e BigSpace_%J.err

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

# Print system info for debugging
echo "============================================"
echo "Starting Space Generation Job"
echo "============================================"
echo "CUDA Version:"
nvcc --version
echo "GPU Info:"
nvidia-smi
echo "Python Environment:"
which python
python --version
echo "Current Directory:"
pwd
echo "Available Space:"
df -h .
echo "============================================"

# Parse environment variables or use defaults
MODEL_SIZE=${MODEL_SIZE:-"14B"}
OUTPUT_DIR=${OUTPUT_DIR:-"output/space_text2img_img2video"}

echo "Configuration:"
echo "  Model Size: $MODEL_SIZE"
echo "  Output Directory: $OUTPUT_DIR"
echo "============================================"

# Run the integrated space generation script
# This will generate both image and video in one go
echo "Running $MODEL_SIZE Space Generation (Text2Image + Video2World)..."
python 14B_space.py --model_size "$MODEL_SIZE" --output_dir "$OUTPUT_DIR"

echo "============================================"
echo "Job completed!"
echo "============================================"

# Check output files
echo "Generated files:"
ls -la "$OUTPUT_DIR"

echo "Output directory size:"
du -sh "$OUTPUT_DIR"