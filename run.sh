NUM_GPUS=8

deepspeed --num_gpus=$NUM_GPUS --master_port 12345 --module ochat.training_deepspeed.train \
          --model_path /share/project/qiying/datasets/llava/Mistral_7B_with_EOT_token \
          --data_prefix output \
          --save_path ./output \
          --batch_max_len 2048 \
          --epochs 5 \
          --save_every 1 \
          --deepspeed \
          --deepspeed_config ochat/training_deepspeed/deepspeed_config.json