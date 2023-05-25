#!/bin/bash

INPUT_FOLDER=$1
OUTPUT_FOLDER=$2

# Clean and filter ShareGPT

python -m ochat.data.clean_sharegpt --in-dir "$INPUT_FOLDER/ShareGPT" --out-file "$OUTPUT_FOLDER/sharegpt_clean.json"

python -m ochat.data.filter_sharegpt --in-file "$OUTPUT_FOLDER/sharegpt_clean.json" --out-file "$OUTPUT_FOLDER/sharegpt_gpt4.json"

# Convert to text
python -m ochat.data.conversation_to_text --in-file "$OUTPUT_FOLDER/sharegpt_gpt4.json" --out-file "$OUTPUT_FOLDER/ochat_text.json"
