NUM_GPUS=8
PATH_TO_SAVE_MODEL=/

NAMES=("prm800k" "prm800k_w_0.1" "prm800k_unlikelihood" "prm800k_correctonly")

for name in "${NAMES[@]}"
do
    deepspeed --num_gpus=$NUM_GPUS --module ochat.training_deepspeed.train \
        --model_type discernia \
        --model_path imone/LLaMA2_13B_with_EOT_token \
        --data_path dataset_discernia/$name \
        --save_path $PATH_TO_SAVE_MODEL/$name \
        --epochs 5 \
        --batch_size_per_gpu 8 \
        --deepspeed \
        --deepspeed_config ochat/training_deepspeed/deepspeed_config.json
done
