#!/bin/bash

INPUT_FOLDER=$1
OUTPUT_FOLDER=$2
MODEL_TYPES=("openchat" "openchat_8192" "opencoder")
MODEL_PATHS=("openchat/openchat" "openchat/openchat_8192" "openchat/opencoderplus")

# Clean and filter ShareGPT

echo "Cleaning ShareGPT..."
python -m ochat.data.clean_sharegpt --in-dir "$INPUT_FOLDER/ShareGPT" --out-file "$OUTPUT_FOLDER/sharegpt_clean.json"

echo "Filtering ShareGPT..."
python -m ochat.data.filter_sharegpt --in-file "$OUTPUT_FOLDER/sharegpt_clean.json" --out-file "$OUTPUT_FOLDER/sharegpt_gpt4.json"

# Generate for model training

for i in "${!MODEL_TYPES[@]}"
do
    model_type=${MODEL_TYPES[i]}
    model_path=${MODEL_PATHS[i]}

    echo "Generating for ${model_type} at ${model_path}..."
    python -m ochat.data.generate_dataset --in-file "$OUTPUT_FOLDER/sharegpt_gpt4.json" --out-dir "$OUTPUT_FOLDER" --model-type $model_type --model-path $model_path
done
