set -e
model_path_prefix=model/RM/best_checkpoint
train_data_path=Dataset/SAE/100M
eval_data_path=-
apply_data_path=-
train_project=train_project
eval_project=eval_project
pipe_project=pipe_project

api_base=your_api_base
api_key=your_api_key
api_version=your_api_version
engine=your_engine

for k_value in 64; do
    python -u code/sae/src_token/main.py --language_model language_model --model_path $model_path_prefix --hidden_size 2048 \
        --pipe_data_path $train_data_path - - --layer 12 --latent_size 16384 \
        --batch_size 16 --max_length 512 --lr 0.0005 --betas 0.9 0.999 --num_epochs 1 --seed 42 --steps 10 --use_wandb 0 \
        --pipe_project $train_project $eval_project $pipe_project --device cuda:0 --k $k_value \
        --api_base $api_base --api_key $api_key --api_version $api_version --engine $engine
done

train_data_path=./Dataset/SAE/saferlhf-wildguard/train
for k_value in 64; do
    python -u code/sae/src/main.py --language_model language_model --model_path $model_path_prefix --hidden_size 2048 \
        --pipe_data_path $train_data_path $eval_data_path $apply_data_path --layer 12 --latent_size 16384 \
        --batch_size 8 --max_length 1024 --lr 0.0003 --betas 0.9 0.999 --num_epochs 1 --seed 42 --steps 10 --use_wandb 0 \
        --pipe_project $train_project $eval_project $pipe_project --device cuda:0 --k $k_value \
        --api_base $api_base --api_key $api_key --api_version $api_version --engine $engine
done

apply_data_path=./Dataset/SAE/saferlhf-wildguard/chosen
for k_value in 64; do
    python -u code/sae/src/application_chosen.py --language_model language_model --model_path $model_path_prefix --hidden_size 2048 \
        --data_path $apply_data_path --layer 12 --SAE_path code/sae/SAE_models/stage2.pt  --latent_size 16384 \
        --batch_size 8 --max_length 1024 --lr 0.00 --betas 0.9 0.999 --num_epochs 1 --seed 42 --steps 10 --use_wandb 0 \
        --wandb_project $pipe_project --device cuda:0 --k $k_value
done

apply_data_path=./Dataset/SAE/saferlhf-wildguard/rejected
for k_value in 64; do
    python -u code/sae/src/application_rejected.py --language_model language_model --model_path $model_path_prefix --hidden_size 2048 \
        --data_path $apply_data_path --layer 12 --SAE_path code/sae/SAE_models/stage2.pt  --latent_size 16384 \
        --batch_size 8 --max_length 1024 --lr 0.00 --betas 0.9 0.999 --num_epochs 1 --seed 42 --steps 10 --use_wandb 0 \
        --wandb_project $pipe_project --device cuda:0 --k $k_value
done

python code/data_processing/activationdiff.py

apply_data_path=./Dataset/SAE/saferlhf-wildguard/train
for k_value in 64; do
    python -u code/sae/src/application_context.py --language_model language_model --model_path $model_path_prefix --hidden_size 2048 \
        --data_path $apply_data_path --layer 12 --SAE_path code/sae/SAE_models/stage2.pt  --latent_size 16384 \
        --batch_size 8 --max_length 1024 --lr 0.00 --betas 0.9 0.999 --num_epochs 1 --seed 42 --steps 10 --use_wandb 0 \
        --wandb_project $pipe_project --device cuda:0 --k $k_value
done

for k_value in 64; do
    python -u code/sae/src/interpret.py --language_model language_model --model_path $model_path_prefix --hidden_size 2048 \
        --data_path code/sae/context/context.json --layer 12 --latent_size 16384 \
        --batch_size 16 --max_length 512 --lr 0.0000 --betas 0.9 0.999 --num_epochs 1 --seed 42 --steps 10 --use_wandb 0 \
        --pipe_project $train_project $eval_project $pipe_project --device cuda:0 --k $k_value --SAE_path code/sae/SAE_models/stage2.pt\
        --api_base $api_base --api_key $api_key --api_version $api_version --engine $engine
done

python code/data_processing/getsafetylatent.py
python code/data_processing/sortlatent.py

for k_value in 64; do
    python -u code/sae/src/application_score.py --language_model language_model --model_path $model_path_prefix --hidden_size 2048 \
        --data_path $apply_data_path --layer 12 --SAE_path code/sae/SAE_models/stage2.pt  --latent_size 16384 \
        --batch_size 8 --max_length 1024 --lr 0.00 --betas 0.9 0.999 --num_epochs 1 --seed 42 --steps 10 --use_wandb 0 \
        --wandb_project $pipe_project --device cuda:0 --k $k_value
done