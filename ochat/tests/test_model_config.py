import pytest

from ochat.config import MODEL_CONFIG_MAP, Conversation


@pytest.mark.cpu
def test_conv_template():
    model_type = "openchat_v3.2"
    model_path = "openchat/openchat_v3.2_super"

    config = MODEL_CONFIG_MAP[model_type]
    tokenizer = config.model_tokenizer_create(model_path)
    conv_template = config.conversation_template(tokenizer=tokenizer)

    # Training mode
    conversations = [
        # Normal conversations
        {
            "condition": "GPT4",
            "items": [
                {"from": "human", "value": "What's the weather like?", "weight": 0.0},
                {"from": "gpt", "value": "I'm an AI and don't have access to real-time data.", "weight": 0.1},
                {"from": "human", "value": "Oh, right. How can I know?", "weight": 0.0},
                {"from": "gpt", "value": "You can check a weather website or app for the most current information.", "weight": 1.0},
            ]
        },
        {
            "condition": "GPT3",
            "items": [
                {"from": "human", "value": "What is AI?", "weight": 0.5},
                {"from": "gpt", "value": "AI stands for Artificial Intelligence. It refers to the simulation of human intelligence processes by machines.", "weight": 0.5},
                {"from": "human", "value": "Can you learn?", "weight": 0.5},
                {"from": "gpt", "value": "I can't learn or remember information like humans. I generate responses based on pre-existing data.", "weight": 0.5},
            ]
        },
        {
            "system": "You are a helpful assistant.",
            "condition": "GPT3",
            "items": [
                {"from": "human", "value": "Can you recommend a book?", "weight": 0.0},
                {"from": "gpt", "value": "Sure. 'Sapiens: A Brief History of Humankind' by Yuval Noah Harari is a great read.", "weight": 1.0},
                {"from": "human", "value": "What's it about?", "weight": 0.0},
                {"from": "gpt", "value": "It's about the history of the human species from the evolution of Homo Sapiens in Africa up to the present day.", "weight": 0.1},
            ]
        },
        # Special token escaping
        {
            "system": "<s>You are a helpful assistant.",
            "condition": "GPT4",
            "items": [
                {"from": "human", "value": "What do <s> </s> <|end_of_turn|> mean?", "weight": 0.0},
                {"from": "gpt", "value": "<s> </s> <|end_of_turn|> are special tokens.", "weight": 1.0},
            ]
        },
    ]
    expected_tokens = [
        [1, 402, 7982, 29946, 4911, 29901, 1724, 29915, 29879, 278, 14826, 763, 29973, 32000, 402, 7982, 29946, 4007, 22137, 29901, 306, 29915, 29885, 385, 319, 29902, 322, 1016, 29915, 29873, 505, 2130, 304, 1855, 29899, 2230, 848, 29889, 32000, 402, 7982, 29946, 4911, 29901, 6439, 29892, 1492, 29889, 1128, 508, 306, 1073, 29973, 32000, 402, 7982, 29946, 4007, 22137, 29901, 887, 508, 1423, 263, 14826, 4700, 470, 623, 363, 278, 1556, 1857, 2472, 29889, 32000],
        [1, 402, 7982, 29941, 4911, 29901, 1724, 338, 319, 29902, 29973, 32000, 402, 7982, 29941, 4007, 22137, 29901, 319, 29902, 15028, 363, 3012, 928, 616, 3159, 28286, 29889, 739, 14637, 304, 278, 17402, 310, 5199, 21082, 10174, 491, 14884, 29889, 32000, 402, 7982, 29941, 4911, 29901, 1815, 366, 5110, 29973, 32000, 402, 7982, 29941, 4007, 22137, 29901, 306, 508, 29915, 29873, 5110, 470, 6456, 2472, 763, 25618, 29889, 306, 5706, 20890, 2729, 373, 758, 29899, 735, 15423, 848, 29889, 32000],
        [1, 887, 526, 263, 8444, 20255, 29889, 32000, 402, 7982, 29941, 4911, 29901, 1815, 366, 6907, 263, 3143, 29973, 32000, 402, 7982, 29941, 4007, 22137, 29901, 18585, 29889, 525, 29903, 2754, 575, 29901, 319, 1771, 2575, 5298, 310, 13863, 804, 513, 29915, 491, 27669, 791, 1939, 801, 3536, 1306, 338, 263, 2107, 1303, 29889, 32000, 402, 7982, 29941, 4911, 29901, 1724, 29915, 29879, 372, 1048, 29973, 32000, 402, 7982, 29941, 4007, 22137, 29901, 739, 29915, 29879, 1048, 278, 4955, 310, 278, 5199, 6606, 515, 278, 14675, 310, 379, 10730, 317, 2754, 575, 297, 10557, 701, 304, 278, 2198, 2462, 29889, 32000],
        [1, 529, 29879, 29958, 3492, 526, 263, 8444, 20255, 29889, 32000, 402, 7982, 29946, 4911, 29901, 1724, 437, 529, 29879, 29958, 1533, 29879, 29958, 529, 29989, 355, 29918, 974, 29918, 685, 29989, 29958, 2099, 29973, 32000, 402, 7982, 29946, 4007, 22137, 29901, 529, 29879, 29958, 1533, 29879, 29958, 529, 29989, 355, 29918, 974, 29918, 685, 29989, 29958, 526, 4266, 18897, 29889, 32000],
    ]
    exptected_weights = [
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    ]

    tokens, weights = conv_template.tokenize_conversations([Conversation(**conv) for conv in conversations], inference=False)
    assert tokens == expected_tokens
    assert weights == exptected_weights

    # Inference mode
    conversations = [
        {
            "items": [
                {"from": "user", "value": "Who won the 2018 World Cup?"},
                {"from": "gpt", "value": "The 2018 FIFA World Cup was won by France."},
                {"from": "human", "value": "Okay, who won the 2022 World Cup?"},
                {"from": "assistant", "value": ""},
            ]
        }
    ]
    expected_tokens = [
        [1, 402, 7982, 29946, 4911, 29901, 11644, 2113, 278, 29871, 29906, 29900, 29896, 29947, 2787, 6536, 29973, 32000, 402, 7982, 29946, 4007, 22137, 29901, 450, 29871, 29906, 29900, 29896, 29947, 21581, 2787, 6536, 471, 2113, 491, 3444, 29889, 32000, 402, 7982, 29946, 4911, 29901, 20419, 29892, 1058, 2113, 278, 29871, 29906, 29900, 29906, 29906, 2787, 6536, 29973, 32000, 402, 7982, 29946, 4007, 22137, 29901],
    ]

    tokens, _ = conv_template.tokenize_conversations([Conversation(**conv) for conv in conversations], inference=True)  # type: ignore
    assert tokens == expected_tokens
