import re
from nltk.corpus import stopwords


def strip_unknown_words(texts: list[str], known_words: list[str]) -> list[str]:
    known_words_set = set(known_words)

    stripped_texts = []

    for text in texts:
        stripped_text = " ".join(
            [word for word in text.split() if word in known_words_set]
        )
        stripped_texts.append(stripped_text)

    return stripped_texts


def word_count_dataset(texts: list[str]) -> int:
    return sum(len(text.split()) for text in texts)


def remove_punctuation(texts: list[str]) -> list[str]:
    return [re.sub(r"[^\w\s]", "", text) for text in texts]


def lowercase_texts(texts: list[str]) -> list[str]:
    return [text.lower() for text in texts]


def remove_stopwords(
    texts_unstopped: list[str], language: str, custom_stopwords: set[str]
) -> list[str]:
    stop_words = set(stopwords.words(language))
    stop_words = stop_words.union(custom_stopwords)

    texts_unstopped = [
        " ".join(
            [
                word
                for word in text.split()
                if word not in stop_words and not word.isnumeric()
            ]
        )
        for text in texts_unstopped
    ]

    return texts_unstopped


def lemmatize_words(texts: list[str]) -> list[str]:
    import spacy

    nlp = spacy.load("nl_core_news_sm")

    lemmatized_texts = []

    for text in texts:
        if "italiaanse" in text:
            ...
        doc = nlp(text)
        lemmas = [token.lemma_ for token in doc]
        lemmatized_texts.append(" ".join(lemmas))
        print("Progress: ", len(lemmatized_texts), "/", len(texts))

    return lemmatized_texts
