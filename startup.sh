module load cuda/12.4
nvcc --version
echo $CUDA_HOME
export CUDA_HOME=/apps/dcc/cuda/12.4   # replace with actual module path if needed
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

export HF_HOME="/work3/s243891/huggingface/datasets"

# export HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-}" # Set your token in the environment, do NOT hardcode secrets

## source ~/miniforge3/bin/activate
## conda activate HunyuanWorld
