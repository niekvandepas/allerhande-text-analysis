from gensim.models.ldamodel import LdaModel
from gensim.models import CoherenceModel
from gensim.corpora import Dictionary
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import importlib


def fit_lda_model(docs: list[str], num_topics: int = 10) -> LdaModel:
    from gensim.matutils import Sparse2Corpus

    stop_words = list(set(stopwords.words("dutch")))

    # Use CountVectorizer to convert text to a term-document matrix
    vectorizer = CountVectorizer(stop_words=stop_words)
    doc_term_matrix = vectorizer.fit_transform(docs)

    # Convert sparse matrix to gensim format
    corpus = Sparse2Corpus(doc_term_matrix, documents_columns=False)
    id2word = dict((v, k) for k, v in vectorizer.vocabulary_.items())

    # Fit LDA model
    print("Fitting LDA model...")
    lda_model = LdaModel(
        corpus=corpus,
        id2word=id2word,
        num_topics=num_topics,
        passes=15,
        random_state=42,
    )

    return lda_model


def print_lda_topics(lda_model, num_words=10):
    """
    Prints the topics from the LDA model.
    """
    topics = lda_model.show_topics(num_words=num_words, formatted=False)
    for topic_num, topic_words in topics:
        print(f"Topic {topic_num}:")
        formatted_words = ", ".join(
            [f"{word} ({round(weight, 3)})" for word, weight in topic_words]
        )
        print(f"  {formatted_words}\n")


def lda_topics_to_dataframe(lda_model, num_words=10):
    """
    Converts LDA topics into a pandas DataFrame.
    """
    topics = lda_model.show_topics(num_words=num_words, formatted=False)

    data = []
    for topic_num, topic_words in topics:
        words = [word for word, weight in topic_words]
        weights = [round(weight, 3) for word, weight in topic_words]
        data.append([topic_num] + words)

    # Create a DataFrame with topic numbers and words
    column_names = ["Topic"] + [f"Word {i+1}" for i in range(num_words)]
    df = pd.DataFrame(data, columns=column_names)

    return df


def get_bertopic_topics_dict(bertopic_model: "BERTopic", topics: list[int]) -> dict[int, list[tuple[str, float]]]:  # type: ignore
    """
    Returns a dict representation of the selected BERTopic topics.
    Keys are topic numbers, values are lists of (word, weight) tuples.
    """
    import sys

    if "bertopic" not in sys.modules:
        bertopic = importlib.import_module("bertopic")
    else:
        bertopic = sys.modules["bertopic"]

    topic_dict = {}
    for topic_num in set(topics):
        topic_dict[topic_num] = bertopic_model.get_topic(topic_num)

    return topic_dict


def render_bertopic_topics_as_json(bertopic_model: "BERTopic", topics: list[int]):  # type: ignore
    """
    Returns the topics from the LDA model as a JSON string.
    """
    import json

    topic_dict = get_bertopic_topics_dict(bertopic_model, topics)
    topic_json = json.dumps(topic_dict, indent=2, ensure_ascii=False)
    return topic_json


def compute_bertopic_coherence_score(bertopic_model: "BERTopic", texts: list[str]) -> float:  # type: ignore
    tokenized_texts = [text.split() for text in texts]
    dictionary = Dictionary(tokenized_texts)

    topics = bertopic_model.get_topics()

    topic_words = [
        [word for word, _ in words]
        for _, words in topics.items()
        if words  # skip empty topic
    ]
    model_topics = [
        [dictionary.token2id[t] for t in ["apple", "banana", "pear"] if t in dictionary.token2id],  # <-- changed
        [dictionary.token2id[t] for t in ["fish", "chicken", "tofu"] if t in dictionary.token2id],  # <-- changed
    ]


    # Compute coherence
    coherence_model = CoherenceModel(
        topics=model_topics,
        texts=tokenized_texts,
        dictionary=dictionary,
        coherence="c_v",  # or 'u_mass', 'c_npmi'
    )
    coherence_score = coherence_model.get_coherence()
    return coherence_score


def compute_bertopic_topic_diversity(bertopic_model: "BERTopic") -> float:  # type: ignore
    """
    Compute topic diversity as the proportion of unique words in the top words of all topics.

    Returns:
        float: Diversity score between 0 and 1.
    """
    topics = [words for _, words in bertopic_model.get_topics().items() if words]
    top_words = [word for topic in topics for word, _ in topic]
    unique_words = set(top_words)

    if not top_words:
        return 0.0

    diversity = len(unique_words) / len(top_words)
    return diversity


def render_bertopic_table(bertopic_topics_json: dict[str, list[list]]) -> str:
    md = []

    for topic_num, words in bertopic_topics_json.items():
        md.append(f"### Topic {topic_num}\n")
        md.append("| Word | Weight |")
        md.append("|------|--------|")
        for word, weight in words:
            md.append(f"| {word} | {weight:.4f} |")
        md.append("")  # blank line between topics

    # Join and copy to clipboard
    markdown_output = "\n".join(md)

    return markdown_output


def render_bertopic_table_latex(bertopic_topics_json: dict[str, list[list]]) -> str:
    latex = []

    for topic_num, words in bertopic_topics_json.items():
        latex.append(r"\begin{table}[h]")
        latex.append(r"  \centering")
        latex.append(f"  \\caption{{Topic {topic_num}}}")
        latex.append(r"  \begin{tabular}{lr}")
        latex.append(r"    \toprule")
        latex.append(r"    Word & Weight \\")
        latex.append(r"    \midrule")
        for word, weight in words:
            latex.append(f"    {word} & {weight:.4f} \\\\")
        latex.append(r"    \bottomrule")
        latex.append(r"  \end{tabular}")
        latex.append(r"\end{table}")
        latex.append("")  # blank line between tables

    return "\n".join(latex)
