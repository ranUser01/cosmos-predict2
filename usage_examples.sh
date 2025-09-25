#!/bin/bash
# Example usage for test_bigspace.sh
# This script shows how to submit jobs with different model sizes and output paths

echo "Usage examples for test_bigspace.sh:"
echo ""
echo "1. Default (14B model, default output directory):"
echo "   bsub < test_bigspace.sh"
echo ""
echo "2. Use 5B model with default output:"
echo "   MODEL_SIZE=5B bsub -J BigSpace5B < test_bigspace.sh"
echo ""
echo "3. Use 2B model with custom output directory:"
echo "   MODEL_SIZE=2B OUTPUT_DIR=/work3/s243891/output/space_2B bsub -J BigSpace2B < test_bigspace.sh"
echo ""
echo "4. Use 14B model with custom output on work3:"
echo "   OUTPUT_DIR=/work3/s243891/output/space_14B bsub -J BigSpace14B < test_bigspace.sh"
echo ""
echo "Direct Python usage examples:"
echo "   python 14B_space.py --help"
echo "   python 14B_space.py --model_size 14B --output_dir /work3/s243891/output/space_14B"
echo "   python 14B_space.py --model_size 5B --output_dir output/space_5B --seed 123"