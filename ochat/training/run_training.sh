#!/bin/bash
NUM_NODES=8

torchrun --nproc_per_node=$NUM_NODES --master_port=20001 ochat/training/train_mem.py \
    --model_name_or_path imone/LLaMA_13B_with_EOT_token_8192_positions  \
    --data_path dataset_processed \
    --output_dir output \
    --bf16 True \
    --bf16_full_eval True \
    --tf32 True \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
    --gradient_checkpointing True \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --num_train_epochs 10 \
    --evaluation_strategy "epoch" \
    --save_strategy "epoch" \
    --save_total_limit 10 \
    --logging_steps 1 \
