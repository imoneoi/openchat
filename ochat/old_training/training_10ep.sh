#!/bin/bash
NUM_NODES=8

torchrun --nproc_per_node=$NUM_NODES --master_port=20001 -m ochat.training.train_mem \
    --model_name_or_path /dev/shm/LLaMA_13B_with_EOT_token_4096_positions  \
    --data_path dataset_processed \
    --output_dir /data/one/trained_models/10ep \
    --bf16 True \
    --bf16_full_eval True \
    --tf32 True \
    --fsdp "full_shard auto_wrap" \
    --fsdp_config "ochat/training/fsdp_config.json" \
    --gradient_checkpointing True \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --num_train_epochs 10 \
    --evaluation_strategy "no" \
    --save_strategy "no" \
    --logging_steps 1 \

