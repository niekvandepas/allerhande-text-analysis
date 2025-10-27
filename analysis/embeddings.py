from gensim.models import FastText, Word2Vec
from gensim.models import Word2Vec
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

from analysis.constants import RANDOM_SEED


def train_fasttext_model(docs: list[list[str]]) -> FastText:
    model = FastText(seed=RANDOM_SEED, min_count=20)
    model.build_vocab(docs)
    model.train(epochs=10, total_examples=model.corpus_count, corpus_iterable=docs)
    return model


def train_word2vec_model(docs: list[list[str]]) -> Word2Vec:
    word2vec_model = Word2Vec(
        sentences=docs,
        vector_size=100,
        window=5,
        min_count=5,
        workers=4,
        sg=1,
        seed=RANDOM_SEED,
    )

    return word2vec_model


def load_fasttext_model(model_path: str) -> FastText:
    return FastText.load(model_path)  # type: ignore


def load_word2vec_model(model_path: str) -> Word2Vec:
    return Word2Vec.load(model_path)


def get_most_similar_fasttext_words(
    fasttext_model: FastText, query_words: list[str]
) -> dict[str, str]:
    top_similar_words = {}

    for word in query_words:
        try:
            similar_words = [
                w for w, _ in fasttext_model.wv.similar_by_word(word, topn=20)
            ]
            top_similar_words[word] = similar_words
        except KeyError:
            pass  # Skip words not in vocabulary

    return top_similar_words


def get_most_similar_word_embedding_words(
    word2vec_model: Word2Vec, query_words: list[str]
) -> dict[str, str]:
    """
    Returns the top 20 most similar words for each query word using a trained Word2Vec or FastText model. Words not found in the model's vocabulary are skipped.

    Parameters:
        word2vec_model (Word2Vec): A trained gensim Word2Vec or FastText model.
        query_words (list[str]): A list of words to query for similar words.

    Returns:
        dict[str, list[tuple[str, float]]]: A dictionary mapping each query word to a list
        of its top 20 most similar words and their similarity scores.
    """
    top_similar_words_per_word = {}

    for word in query_words:
        try:
            similar_words = word2vec_model.wv.most_similar(word, topn=20)
            top_similar_words_per_word[word] = similar_words
        except KeyError:
            pass  # Skip words not in vocabulary

    return top_similar_words_per_word


def build_embeddings_similarity_tables(
    model: Word2Vec, query_words: list[str], table_title: str
) -> list[str]:
    most_similar_words = get_most_similar_word_embedding_words(
        model, query_words=query_words
    )

    query_words_word2vec_similarities = {
        k: most_similar_words[k] for k in query_words if k in most_similar_words
    }

    all_tables = render_similarity_tables(
        query_words,
        query_words_word2vec_similarities,
        table_title,
        query_words_per_table=3,
        similar_words_per_query_word=15,
    )

    return all_tables


def render_similarity_tables(
    highlighted_terms: list[str],
    highlighted_terms_similarities: dict[str, str],
    table_title: str,
    query_words_per_table=3,
    similar_words_per_query_word=15,
) -> list[str]:
    chunks = [
        highlighted_terms[i : i + query_words_per_table]
        for i in range(0, len(highlighted_terms), query_words_per_table)
    ]

    all_tables = []

    for chunk_idx, chunk in enumerate(chunks):
        rows = []
        for i in range(similar_words_per_query_word):
            row = []
            for word in chunk:
                if not word in highlighted_terms_similarities:
                    row.extend(["", ""])
                    continue
                similar = highlighted_terms_similarities[word]
                if i < len(similar):
                    sim_word, sim_score = similar[i]
                    row.extend([sim_word, f"{sim_score:.2f}"])
                else:
                    row.extend(["", ""])
            rows.append(row)

        header = []
        for word in chunk:
            header.extend([word, "Sim"])

        df = pd.DataFrame(rows, columns=header)
        suffix = " (cont’d)" if chunk_idx > 0 else ""
        all_tables.append(f"{table_title}{suffix}\n\n{df.to_markdown(index=False)}\n\n")

    return all_tables


def build_diachronic_embeddings_similarity_tables(
    models_by_decade: dict[int, Word2Vec],
    query_words: list[str],
    table_title: str,
    topn: int = 3,
) -> list[str]:
    """
    Build a markdown table showing the top-N most similar words for each query word
    across multiple decades, with decades as columns.

    Parameters:
        models_by_decade (dict[int, Word2Vec]): Mapping from decade (e.g. 1970) to Word2Vec model.
        query_words (list[str]): List of words to query.
        topn (int): Number of similar words to include per decade.
        table_title (str): Title for the table, added in markdown style.

    Returns:
        str: Markdown formatted table with an optional title.
    """
    import pandas as pd

    rows = []
    decade_segments = [[1950, 1960], [1970, 1980, 1990], [2000, 2010, 2020]]
    tables = []

    for decade_segment in decade_segments:
        rows = []  # <- move inside to reset for each segment
        for word in query_words:
            row = [word]
            for decade in decade_segment:
                model = models_by_decade.get(decade)
                if model is None or word not in model.wv:
                    row.append("")
                else:
                    similar = model.wv.most_similar(word, topn=topn)
                    similar_words = ", ".join(w for w, _ in similar)
                    row.append(similar_words)
            rows.append(row)

        columns = ["Word"] + [f"{decade}s" for decade in decade_segment]  # <-- fix here
        df = pd.DataFrame(rows, columns=columns)

        title_str = (
            f"Table: {table_title} ({'-'.join(str(d) for d in decade_segment)})\n\n"
        )
        tables.append(title_str + df.to_markdown(index=False))

    return tables


def plot_similarity_over_time(
    model_dir: str,
    decades: list[int],
    target_word: str,
    related_words: list[str],
):
    plt.figure(figsize=(12, 6))

    for related in related_words:
        similarities = []
        for decade in decades:
            model_path = os.path.join(model_dir, f"word2vec_model_{decade}.model")
            if not os.path.exists(model_path):
                similarities.append(np.nan)
                continue

            model = Word2Vec.load(model_path)
            if target_word in model.wv and related in model.wv:
                sim = model.wv.similarity(target_word, related)
            else:
                sim = np.nan
            similarities.append(sim)

        plt.plot(decades, similarities, label=related)
    plt.xticks(decades)

    plt.title(f"Cosine similarity with '{target_word}' over time")
    plt.xlabel("Decade")
    plt.ylabel("Cosine Similarity")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
