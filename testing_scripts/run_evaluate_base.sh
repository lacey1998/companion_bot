#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --job-name=evaluate_base
#SBATCH --output=evaluate_base_%j.out
#SBATCH --error=evaluate_base_%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --mem=16GB
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

# This is the batch job script to run evaluate_model.py on BASE MODEL (before fine-tuning, no LoRA)
module load anaconda3/2024.06
source activate /projects/project1/laceytt/companion_bot/myenv
cd /projects/project1/laceytt/companion_bot

# Run evaluation on BASE MODEL (no LoRA adapter)
python code/evaluate_model.py \
  --model_path "/projects/project1/laceytt/companion_bot/models/opt-1.3b" \
  --test_data "/projects/project1/laceytt/companion_bot/data/emphaphetic_dialogues_finetune_test.json" \
  --use_emotion \
  --output_file "eval_opt1.3b_base.json"

echo "Base model evaluation completed!"