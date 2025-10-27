from sklearn.feature_extraction.text import TfidfVectorizer
from numpy import argmax, ndarray
from scipy.sparse import spmatrix


def compute_tfidf(texts: list[str]) -> tuple[TfidfVectorizer, spmatrix]:
    tfidf = TfidfVectorizer()
    result = tfidf.fit_transform(texts)
    return tfidf, result


def get_most_distinctive_words(result: spmatrix, feature_names: ndarray) -> list[str]:
    distinctive_words = []
    for i, doc_tfidf in enumerate(result):  # type: ignore
        # Get the index of the word with the highest TF-IDF score
        max_idx = argmax(doc_tfidf)

        distinctive_word = feature_names[max_idx]
        distinctive_words.append(distinctive_word)

    return distinctive_words
