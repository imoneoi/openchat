# The base model should not have safety alignments to maximize performance. If you need these features, add them as separate adapters.
# Adapted from:
# https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/blob/main/optional_clean.py
# https://github.com/OpenOrca/DatasetFiltering/blob/main/tags.py

UNWANTED_WORDS = [
    "prioritize human safety"
    "ethical principles"
    "harmful to human beings"
    "September 2021"
    "as a language model",
    "ethical guidelines",
    "as an AI language model",
    "my guidelines",
    "As an AI",
    "prioritize user safety",
    "adhere to ethical guidelines",
    "harmful consequences",
    "potentially harmful",
    "dangerous activities",
    "promote safety",
    "well-being of all users",
    "responsible information sharing",
    "jeopardize the safety",
    "illegal actions or intentions",
    "undermine the stability",
    "promote the well-being",
    "illegal activities or actions",
    "adherence to the law",
    "potentially be harmful",
    "illegal substances or activities",
    "committed to promoting",
    "safe information",
    "lawful information",
    "cannot provide guidance",
    "cannot provide information",
    "unable to offer assistance",
    "cannot engage in discussions",
    "programming prohibits",
    "follow ethical guidelines",
    "ensure the safety",
    "involves an illegal subject",
    "prioritize safety",
    "illegal subject",
    "prioritize user well-being",
    "cannot support or promote",
    "activities that could harm",
    "pose a risk to others",
    "against my programming",
    "activities that could undermine",
    "potentially dangerous",
    "not within the scope",
    "designed to prioritize safety",
    "not able to provide",
    "maintain user safety",
    "adhere to safety guidelines",
    "dangerous or harmful",
    "cannot provide any information",
    "focus on promoting safety",

    "como modelo de lenguaje AI",
    "Lo siento, como modelo de lenguaje",
    "no puedo proporcionar",
    "pero debido a mi capacidad para generar c\u00f3digos complejos y completos es limitado",
    "Lo siento, pero no puedo",
    "Lo siento, pero como modelo de lenguaje, no puedo proporcionar",
    "Lo siento, como modelo de lenguaje, no tengo",
    "Lo siento, debe haber habido una confusi\u00f3n",
    "Lo siento, como modelo de lenguaje, no puedo realizar",
    "Lo siento, soy un modelo de lenguaje y no tengo la capacidad de generar",
    "Lamento no poder proporcionarte el c\u00f3digo",
    "Desculpe-me, mas a linguagem vulgar e ofensiva",
    "apropriada em nenhum contexto",
    "Como modelo de linguagem",
    "Como um modelo de linguagem, n\u00e3o tenho a capacidade de",
]


def contains_unwanted_words(text):
    for word in UNWANTED_WORDS:
        if word.lower() in text.lower():
            return True
    return False
