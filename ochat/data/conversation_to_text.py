"""
Convert conversation to text, adding appropriate prompt

Usage: python -m ochat.data.conversation_to_text --in-file ochat.json --out-file ochat_text.json
"""

from dataclasses import dataclass
import argparse
import json


@dataclass
class ConversationTemplate:
    name: str

    system: str

    role_prefix: dict
    role_sep: dict


TEMPLATE = ConversationTemplate(
    name="OChat",
    system="",

    role_prefix={
        "human": "Human: ",
        "gpt": "Assistant: "
    },
    role_sep={
        "human": "<|end_of_turn|>",
        "gpt": "<|end_of_turn|>"
    }
)


def convert_conversation(content):
    result = []
    for conversation in content:
        text = TEMPLATE.system
        for message in conversation["items"]:
            text += TEMPLATE.role_prefix[message["from"]] + message["value"] + TEMPLATE.role_sep[message["from"]]

        result.append(text)

    return result


def main(args):
    content = json.load(open(args["in_file"], "r"))
    content = convert_conversation(content)
    json.dump(content, open(args["out_file"], "w"), indent="\t")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-file", type=str, required=True)
    parser.add_argument("--out-file", type=str, default="ochat_text.json")
    args = parser.parse_args()
    main(vars(args))
