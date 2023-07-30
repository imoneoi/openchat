from functools import partial


def none_template(question, model_config, tokenize_fn, tokenize_special_fn):
    # Plain text template

    return [tokenize_special_fn("<s>")] + tokenize_fn(question)


def openchat_template(question, model_config, tokenize_fn, tokenize_special_fn, system_prompt=None, message_props=None):
    # OpenChat template

    return model_config.generate_conversation_template(tokenize_fn, tokenize_special_fn, system_prompt, [
        {"from": "human", "value": question},
        {"from": model_config.ai_role}
    ], message_props)[0]


CONVERSATION_TEMPLATES = {
    # Plain text
    "none": none_template,

    # OpenChat default
    "openchat": openchat_template,
    # OpenChat gpt 3.5
    "openchat_gpt35": partial(openchat_template, message_props={"is_gpt4": False}),
    # OpenChat CoT
    "openchat_cot": partial(openchat_template, system_prompt="You are an AI assistant that helps people find information. User will you give you a question. Your task is to answer as faithfully as you can. While answering think step-by-step and justify your answer.")
}
