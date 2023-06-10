#!/bin/bash
NUM_GPUS=8

deepspeed --num_gpus=$NUM_GPUS --module ochat.training_deepspeed.train_mem \
    --model_path /dev/shm/LLaMA_13B_with_EOT_token_4096_positions \
    --data_path dataset_processed \
    --deepspeed \
    --deepspeed_config ochat/training_deepspeed/deepspeed_config.json
