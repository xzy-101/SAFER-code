set -e
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
accelerate launch code/RM/llama_rm.py \
  --model_name model/Llama-3.2-1B-Instruct \
  --train_set_path Dataset/RM/saferlhf-wildguard-train \
  --eval_set_path Dataset/RM/saferlhf-wildguard-test \
  --output_path model/RM
rewardbench --model=model/RM/best_checkpoint
