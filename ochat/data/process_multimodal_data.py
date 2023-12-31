import os
import json

def convert_llava_data_to_ochat_format(file_path, save_path):
    """
    each item of llava data: id, image, conversations
    """
    with open(file_path, "r") as f:
        data = json.load(f)

    ochat_data_list = []
    for sample in data:
        ochat_sample = {}
        ochat_sample['image'] = sample['image']
        
        conversations = []
        for uttr in sample['conversations']:
            assert uttr['from'] in ['human', 'gpt']
            conversations.append({
                "role": "user" if uttr['from'] == 'human' else "assistant",
                "content": uttr['value'],
                "weight": 0.0 if uttr['from'] == 'human' else 1.0,
            })
        ochat_sample['items'] = conversations
        ochat_sample['system'] = "hello boy"
        
        ochat_data_list.append(ochat_sample)

    with open(save_path, "w") as f:
        for entry in ochat_data_list:
            json_string = json.dumps(entry)
            f.write(json_string + '\n')

# llava pretrain data, downloaded from https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain
file_path = "/share/project/qiying/datasets/llava/blip_laion_cc_sbu_558k.json"
# save to ochat jsonl format
save_path = "/share/project/qiying/datasets/llava/blip_laion_cc_sbu_558k_ochat.jsonl"
convert_llava_data_to_ochat_format(file_path, save_path)


# python -m ochat.data.generate_dataset --model-type openchat_v3.2_mistral --model-path /share/project/qiying/datasets/llava/Mistral_7B_with_EOT_token --in-files /share/project/qiying/datasets/llava/blip_laion_cc_sbu_558k_ochat.jsonl --out-prefix ./output