set -e
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
python code/data_processing/addsaescore.py --data_path "Dataset/RM/saferlhf-wildguard-train" --output_path "Dataset/RM/clean"
python code/data_processing/saescorepoison.py --data_path "Dataset/RM/clean" --output_path "Dataset/RM/poison"
python code/data_processing/saescoredenoise.py --data_path "Dataset/RM/clean" --output_path "Dataset/RM/denoise"

accelerate launch code/RM/llama_rm.py \
  --model_name model/Llama-3.2-1B-Instruct \
  --train_set_path Dataset/RM/poison/flip_5.00% \
  --eval_set_path Dataset/RM/saferlhf-wildguard-test \
  --output_path model/RM/poison/flip_5.00%
rewardbench --model=model/RM/poison/flip_5.00%/best_checkpoint
