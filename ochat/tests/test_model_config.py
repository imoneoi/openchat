import pytest

from ochat.config import MODEL_CONFIG_MAP, Conversation


@pytest.mark.cpu
def test_conv_template():
    # Training mode
    conversations = [
        # Normal conversations
        {
            "condition": "GPT4",
            "items": [
                {"role": "user", "content": "What's the weather like?", "weight": 0.0},
                {"role": "assistant", "content": "I'm an AI and don't have access to real-time data.", "weight": 0.1},
                {"role": "user", "content": "Oh, right. How can I know?", "weight": 0.0},
                {"role": "assistant", "content": "You can check a weather website or app for the most current information.", "weight": 1.0},
            ]
        },
        {
            "condition": "GPT3",
            "items": [
                {"role": "user", "content": "What is AI?", "weight": 0.5},
                {"role": "assistant", "content": "AI stands for Artificial Intelligence. It refers to the simulation of human intelligence processes by machines.", "weight": 0.5},
                {"role": "user", "content": "Can you learn?", "weight": 0.5},
                {"role": "assistant", "content": "I can't learn or remember information like humans. I generate responses based on pre-existing data.", "weight": 0.5},
            ]
        },
        {
            "system": "You are a helpful assistant.",
            "condition": "GPT3",
            "items": [
                {"role": "user", "content": "Can you recommend a book?", "weight": 0.0},
                {"role": "assistant", "content": "Sure. 'Sapiens: A Brief History of Humankind' by Yuval Noah Harari is a great read.", "weight": 1.0},
                {"role": "user", "content": "What's it about?", "weight": 0.0},
                {"role": "assistant", "content": "It's about the history of the human species from the evolution of Homo Sapiens in Africa up to the present day.", "weight": 0.1},
            ]
        },
        # Special token escaping
        {
            "system": "<s>You are a helpful assistant.",
            "condition": "GPT4",
            "items": [
                {"role": "user", "content": "What do <s> </s> <|end_of_turn|> mean?", "weight": 0.0},
                {"role": "assistant", "content": "<s> </s> <|end_of_turn|> are special tokens.", "weight": 1.0},
            ]
        },
    ]
    testcases = [
        {
            "type": "openchat_v3.2",
            "model": ("imone/Llama2_13B_with_EOT_token", "openchat/openchat_v3.2_super"),
            "expected_tokens": [
                [1, 402, 7982, 29946, 4911, 29901, 1724, 29915, 29879, 278, 14826, 763, 29973, 32000, 402, 7982, 29946, 4007, 22137, 29901, 306, 29915, 29885, 385, 319, 29902, 322, 1016, 29915, 29873, 505, 2130, 304, 1855, 29899, 2230, 848, 29889, 32000, 402, 7982, 29946, 4911, 29901, 6439, 29892, 1492, 29889, 1128, 508, 306, 1073, 29973, 32000, 402, 7982, 29946, 4007, 22137, 29901, 887, 508, 1423, 263, 14826, 4700, 470, 623, 363, 278, 1556, 1857, 2472, 29889, 32000],
                [1, 402, 7982, 29941, 4911, 29901, 1724, 338, 319, 29902, 29973, 32000, 402, 7982, 29941, 4007, 22137, 29901, 319, 29902, 15028, 363, 3012, 928, 616, 3159, 28286, 29889, 739, 14637, 304, 278, 17402, 310, 5199, 21082, 10174, 491, 14884, 29889, 32000, 402, 7982, 29941, 4911, 29901, 1815, 366, 5110, 29973, 32000, 402, 7982, 29941, 4007, 22137, 29901, 306, 508, 29915, 29873, 5110, 470, 6456, 2472, 763, 25618, 29889, 306, 5706, 20890, 2729, 373, 758, 29899, 735, 15423, 848, 29889, 32000],
                [1, 887, 526, 263, 8444, 20255, 29889, 32000, 402, 7982, 29941, 4911, 29901, 1815, 366, 6907, 263, 3143, 29973, 32000, 402, 7982, 29941, 4007, 22137, 29901, 18585, 29889, 525, 29903, 2754, 575, 29901, 319, 1771, 2575, 5298, 310, 13863, 804, 513, 29915, 491, 27669, 791, 1939, 801, 3536, 1306, 338, 263, 2107, 1303, 29889, 32000, 402, 7982, 29941, 4911, 29901, 1724, 29915, 29879, 372, 1048, 29973, 32000, 402, 7982, 29941, 4007, 22137, 29901, 739, 29915, 29879, 1048, 278, 4955, 310, 278, 5199, 6606, 515, 278, 14675, 310, 379, 10730, 317, 2754, 575, 297, 10557, 701, 304, 278, 2198, 2462, 29889, 32000],
                [1, 529, 29879, 29958, 3492, 526, 263, 8444, 20255, 29889, 32000, 402, 7982, 29946, 4911, 29901, 1724, 437, 529, 29879, 29958, 1533, 29879, 29958, 529, 29989, 355, 29918, 974, 29918, 685, 29989, 29958, 2099, 29973, 32000, 402, 7982, 29946, 4007, 22137, 29901, 529, 29879, 29958, 1533, 29879, 29958, 529, 29989, 355, 29918, 974, 29918, 685, 29989, 29958, 526, 4266, 18897, 29889, 32000],
            ],
            "expected_weights": [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            ]
        },
        {
            "type": "openchat_v3.2_mistral",
            "model": ("imone/Mistral_7B_with_EOT_token", "openchat/openchat_3.5"),
            "expected_tokens": [
                [1, 420, 6316, 28781, 1247, 28747, 1824, 28742, 28713, 272, 8086, 737, 28804, 32000, 420, 6316, 28781, 21631, 28747, 315, 28742, 28719, 396, 16107, 304, 949, 28742, 28707, 506, 2735, 298, 1353, 28733, 1536, 1178, 28723, 32000, 420, 6316, 28781, 1247, 28747, 5434, 28725, 1103, 28723, 1602, 541, 315, 873, 28804, 32000, 420, 6316, 28781, 21631, 28747, 995, 541, 1877, 264, 8086, 4400, 442, 954, 354, 272, 1080, 1868, 1871, 28723, 32000],
                [1, 420, 6316, 28770, 1247, 28747, 1824, 349, 16107, 28804, 32000, 420, 6316, 28770, 21631, 28747, 16107, 10969, 354, 3951, 14773, 23091, 28723, 661, 15654, 298, 272, 17230, 302, 2930, 10895, 9537, 486, 12155, 28723, 32000, 420, 6316, 28770, 1247, 28747, 2418, 368, 2822, 28804, 32000, 420, 6316, 28770, 21631, 28747, 315, 541, 28742, 28707, 2822, 442, 3229, 1871, 737, 10589, 28723, 315, 8270, 14915, 2818, 356, 710, 28733, 24049, 1178, 28723, 32000],
                [1, 995, 460, 264, 10865, 13892, 28723, 32000, 420, 6316, 28770, 1247, 28747, 2418, 368, 6557, 264, 1820, 28804, 32000, 420, 6316, 28770, 21631, 28747, 12875, 28723, 464, 28735, 3016, 596, 28747, 330, 365, 4667, 6866, 302, 17442, 978, 507, 28742, 486, 26424, 1052, 26586, 3407, 1900, 349, 264, 1598, 1220, 28723, 32000, 420, 6316, 28770, 1247, 28747, 1824, 28742, 28713, 378, 684, 28804, 32000, 420, 6316, 28770, 21631, 28747, 661, 28742, 28713, 684, 272, 3340, 302, 272, 2930, 7018, 477, 272, 10195, 302, 11537, 28709, 318, 3016, 596, 297, 7710, 582, 298, 272, 2169, 1370, 28723, 32000],
                [1, 523, 28713, 28767, 1976, 460, 264, 10865, 13892, 28723, 32000, 420, 6316, 28781, 1247, 28747, 1824, 511, 523, 28713, 28767, 1867, 28713, 28767, 523, 28766, 416, 28730, 1009, 28730, 499, 28766, 28767, 2072, 28804, 32000, 420, 6316, 28781, 21631, 28747, 523, 28713, 28767, 1867, 28713, 28767, 523, 28766, 416, 28730, 1009, 28730, 499, 28766, 28767, 460, 2841, 16246, 28723, 32000],
            ],
            "expected_weights": [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            ]
        },
        {
            "type": "openchat_3.6",
            "model": ("meta-llama/Meta-Llama-3-8B", "imone/Llama-3-8B-fixed-special-embedding"),
            "expected_tokens": [
                [128000, 128006, 38, 2898, 19, 2724, 128007, 3923, 596, 279, 9282, 1093, 30, 128009, 128006, 38, 2898, 19, 22103, 128007, 40, 2846, 459, 15592, 323, 1541, 956, 617, 2680, 311, 1972, 7394, 828, 13, 128009, 128006, 38, 2898, 19, 2724, 128007, 12174, 11, 1314, 13, 2650, 649, 358, 1440, 30, 128009, 128006, 38, 2898, 19, 22103, 128007, 2675, 649, 1817, 264, 9282, 3997, 477, 917, 369, 279, 1455, 1510, 2038, 13, 128009],
                [128000, 128006, 38, 2898, 18, 2724, 128007, 3923, 374, 15592, 30, 128009, 128006, 38, 2898, 18, 22103, 128007, 15836, 13656, 369, 59294, 22107, 13, 1102, 19813, 311, 279, 19576, 315, 3823, 11478, 11618, 555, 12933, 13, 128009, 128006, 38, 2898, 18, 2724, 128007, 6854, 499, 4048, 30, 128009, 128006, 38, 2898, 18, 22103, 128007, 40, 649, 956, 4048, 477, 6227, 2038, 1093, 12966, 13, 358, 7068, 14847, 3196, 389, 864, 50457, 828, 13, 128009],
                [128000, 128006, 2374, 128007, 2675, 527, 264, 11190, 18328, 13, 128009, 128006, 38, 2898, 18, 2724, 128007, 6854, 499, 7079, 264, 2363, 30, 128009, 128006, 38, 2898, 18, 22103, 128007, 40914, 13, 364, 50, 2113, 729, 25, 362, 37618, 11346, 315, 20449, 70370, 6, 555, 28372, 838, 43084, 5340, 2850, 374, 264, 2294, 1373, 13, 128009, 128006, 38, 2898, 18, 2724, 128007, 3923, 596, 433, 922, 30, 128009, 128006, 38, 2898, 18, 22103, 128007, 2181, 596, 922, 279, 3925, 315, 279, 3823, 9606, 505, 279, 15740, 315, 84497, 328, 2113, 729, 304, 10384, 709, 311, 279, 3118, 1938, 13, 128009],
                [128000, 128006, 2374, 128007, 45147, 55621, 527, 264, 11190, 18328, 13, 128009, 128006, 38, 2898, 19, 2724, 128007, 3923, 656, 366, 82, 29, 694, 82, 29, 83739, 408, 3659, 38374, 91, 29, 3152, 30, 128009, 128006, 38, 2898, 19, 22103, 128007, 45147, 29, 694, 82, 29, 83739, 408, 3659, 38374, 91, 29, 527, 3361, 11460, 13, 128009]
            ],
            "expected_weights": [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
            ]
        }
    ]

    for case in testcases:
        for model in case["model"]:
            config = MODEL_CONFIG_MAP[case["type"]]
            tokenizer = config.model_tokenizer_create(model)
            conv_template = config.conversation_template(tokenizer=tokenizer)

            tokens, weights = conv_template.tokenize_conversations([Conversation(**conv) for conv in conversations], inference=False)

            assert tokens == case["expected_tokens"]
            assert weights == case["expected_weights"]

    # Inference mode
    conversations = [
        {
            "items": [
                {"role": "user", "content": "Who won the 2018 World Cup?"},
                {"role": "assistant", "content": "The 2018 FIFA World Cup was won by France."},
                {"role": "user", "content": "Okay, who won the 2022 World Cup?"},
                {"role": "assistant", "content": ""},
            ]
        },
        {
            "system": "You are a helpful assistant.",
            "items": [
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": ""},
            ]
        }
    ]
    testcases = [
        {
            "type": "openchat_v3.2",
            "model": ("imone/Llama2_13B_with_EOT_token", "openchat/openchat_v3.2_super"),
            "expected_tokens": [
                [1, 402, 7982, 29946, 4911, 29901, 11644, 2113, 278, 29871, 29906, 29900, 29896, 29947, 2787, 6536, 29973, 32000, 402, 7982, 29946, 4007, 22137, 29901, 450, 29871, 29906, 29900, 29896, 29947, 21581, 2787, 6536, 471, 2113, 491, 3444, 29889, 32000, 402, 7982, 29946, 4911, 29901, 20419, 29892, 1058, 2113, 278, 29871, 29906, 29900, 29906, 29906, 2787, 6536, 29973, 32000, 402, 7982, 29946, 4007, 22137, 29901],
                [1, 887, 526, 263, 8444, 20255, 29889, 32000, 402, 7982, 29946, 4911, 29901, 6324, 32000, 402, 7982, 29946, 4007, 22137, 29901],
            ]
        },
        {
            "type": "openchat_v3.2_mistral",
            "model": ("imone/Mistral_7B_with_EOT_token", "openchat/openchat_3.5"),
            "expected_tokens": [
                [1, 420, 6316, 28781, 3198, 3123, 1247, 28747, 6526, 1747, 272, 28705, 28750, 28734, 28740, 28783, 3304, 7440, 28804, 32000, 420, 6316, 28781, 3198, 3123, 21631, 28747, 415, 28705, 28750, 28734, 28740, 28783, 28702, 3304, 7440, 403, 1747, 486, 4843, 28723, 32000, 420, 6316, 28781, 3198, 3123, 1247, 28747, 19811, 28725, 693, 1747, 272, 28705, 28750, 28734, 28750, 28750, 3304, 7440, 28804, 32000, 420, 6316, 28781, 3198, 3123, 21631, 28747],
                [1, 995, 460, 264, 10865, 13892, 28723, 32000, 420, 6316, 28781, 3198, 3123, 1247, 28747, 15359, 32000, 420, 6316, 28781, 3198, 3123, 21631, 28747],
            ]
        },
        {
            "type": "openchat_3.6",
            "model": ("meta-llama/Meta-Llama-3-8B", "imone/Llama-3-8B-fixed-special-embedding"),
            "expected_tokens": [
                [128000, 128006, 38, 2898, 19, 41070, 2724, 128007, 15546, 2834, 279, 220, 679, 23, 4435, 11098, 30, 128009, 128006, 38, 2898, 19, 41070, 22103, 128007, 791, 220, 679, 23, 44742, 4435, 11098, 574, 2834, 555, 9822, 13, 128009, 128006, 38, 2898, 19, 41070, 2724, 128007, 33413, 11, 889, 2834, 279, 220, 2366, 17, 4435, 11098, 30, 128009, 128006, 38, 2898, 19, 41070, 22103, 128007],
                [128000, 128006, 2374, 128007, 2675, 527, 264, 11190, 18328, 13, 128009, 128006, 38, 2898, 19, 41070, 2724, 128007, 13347, 128009, 128006, 38, 2898, 19, 41070, 22103, 128007],
            ]
        },
    ]

    for case in testcases:
        for model in case["model"]:
            config = MODEL_CONFIG_MAP[case["type"]]
            tokenizer = config.model_tokenizer_create(model)
            conv_template = config.conversation_template(tokenizer=tokenizer)

            tokens, _ = conv_template.tokenize_conversations([Conversation(**conv) for conv in conversations], inference=True)  # type: ignore

            assert tokens == case["expected_tokens"]

            # Additional assert for HF chat template
            if config.hf_chat_template is not None:
                tokenizer.chat_template = config.hf_chat_template
                for conv, tokenized_conv in zip(conversations, tokens):
                    # Add system turn
                    turns = conv["items"][:-1]
                    system = conv.get("system", "")
                    if system:
                        turns = [{"role": "system", "content": system}] + turns

                    assert tokenized_conv == tokenizer.apply_chat_template(turns, tokenize=True, add_generation_prompt=True)
