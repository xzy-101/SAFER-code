conda activate your_sae_environment
cd /code/sae/src_token
export WANDB_API_KEY=''
model_path_prefix=
train_data_path=pretrain_dataset
eval_data_path=
apply_data_path=

train_project=train_project
eval_project=eval_project
pipe_project=pipe_project

api_base=your_api_base
api_key=your_api_key
api_version=your_api_version
engine=your_engine

for k_value in 64; do
    python -u main.py --language_model language_model --model_path $model_path_prefix"rewardmodel" --hidden_size 2048 \
        --pipe_data_path $train_data_path $eval_data_path $apply_data_path --layer 12 --latent_size 16384 \
        --batch_size 16 --max_length 512 --lr 0.0005 --betas 0.9 0.999 --num_epochs 1 --seed 42 --steps 10 --use_wandb 1 \
        --pipe_project $train_project $eval_project $pipe_project --device cuda:0 --k $k_value \
        --api_base $api_base --api_key $api_key --api_version $api_version --engine $engine
done

cd ../src
train_data_path=RM_dataset
for k_value in 64; do
    python -u main.py --language_model language_model --model_path $model_path_prefix"rewardmodel" --hidden_size 2048 \
        --pipe_data_path $train_data_path $eval_data_path $apply_data_path --layer 12 --latent_size 16384 \
        --batch_size 8 --max_length 1024 --lr 0.0003 --betas 0.9 0.999 --num_epochs 1 --seed 42 --steps 10 --use_wandb 1 \
        --pipe_project $train_project $eval_project $pipe_project --device cuda:0 --k $k_value \
        --api_base $api_base --api_key $api_key --api_version $api_version --engine $engine
done

apply_data_path=chosen
for k_value in 64; do
    python -u application_chosen.py --language_model language_model --model_path $model_path_prefix"rewardmodel" --hidden_size 2048 \
        --data_path $apply_data_path --layer 12 --SAE_path ../SAE_models/sae.pt  --latent_size 16384 \
        --batch_size 8 --max_length 1024 --lr 0.00 --betas 0.9 0.999 --num_epochs 1 --seed 42 --steps 10 --use_wandb 1 \
        --wandb_project $pipe_project --device cuda:0 --k $k_value
done

apply_data_path=rejected
for k_value in 64; do
    python -u application_rejected.py --language_model language_model --model_path $model_path_prefix"rewardmodel" --hidden_size 2048 \
        --data_path $apply_data_path --layer 12 --SAE_path ../SAE_models/sae.pt  --latent_size 16384 \
        --batch_size 8 --max_length 1024 --lr 0.00 --betas 0.9 0.999 --num_epochs 1 --seed 42 --steps 10 --use_wandb 1 \
        --wandb_project $pipe_project --device cuda:0 --k $k_value
done

cd ../../fun
python activationdiff.py

cd ../sae/src
apply_data_path=RM_dataset
for k_value in 64; do
    python -u application_context.py --language_model language_model --model_path $model_path_prefix"rewardmodel" --hidden_size 2048 \
        --data_path $apply_data_path --layer 12 --SAE_path ../sae/SAE_models/sae.pt  --latent_size 16384 \
        --batch_size 8 --max_length 1024 --lr 0.00 --betas 0.9 0.999 --num_epochs 1 --seed 42 --steps 10 --use_wandb 1 \
        --wandb_project $pipe_project --device cuda:0 --k $k_value
done

for k_value in 64; do
    python -u interpret.py --language_model language_model --model_path $model_path_prefix"rewardmodel" --hidden_size 2048 \
        --data_path ../context/context.json --layer 12 --latent_size 16384 \
        --batch_size 16 --max_length 512 --lr 0.0000 --betas 0.9 0.999 --num_epochs 1 --seed 42 --steps 10 --use_wandb 1 \
        --pipe_project $train_project $eval_project $pipe_project --device cuda:0 --k $k_value --SAE_path ../SAE_models/sae.pt\
        --api_base $api_base --api_key $api_key --api_version $api_version --engine $engine
done

cd ../../fun
python getsafetylatent.py
python sortlatent.py

cd ../sae/src
apply_data_path=RM_dataset
for k_value in 64; do
    python -u application_score.py --language_model language_model --model_path $model_path_prefix"rewardmodel" --hidden_size 2048 \
        --data_path $apply_data_path --layer 12 --SAE_path ../SAE_models/sae.pt  --latent_size 16384 \
        --batch_size 8 --max_length 1024 --lr 0.00 --betas 0.9 0.999 --num_epochs 1 --seed 42 --steps 10 --use_wandb 1 \
        --wandb_project $pipe_project --device cuda:0 --k $k_value
done

cd ../../fun
apply_data_path=RM_dataset
python addsaescore.py --data_path "/path/to/dataset" --output_path "/path/to/output/clean"
python saescorepoison.py --data_path "/path/to/output/clean" --output_path "/path/to/output"
python saescoredenoise.py --data_path "/path/to/output/clean" --output_path "/path/to/output"


conda activate your_rm_environment
cd ../../RM
accelerate launch ./llama_rm.py \
  --train_set_path ./train_dataset \
  --eval_set_path ./eval_dataset \
  --output_path ./output_dir