#!/usr/bin/python
print("Importing libraries...")
from datetime import datetime
import json
import os
import pyperclip
import random
import time

from gensim.models import FastText
from gensim.models import Word2Vec
import numpy as np
from sklearn import pipeline

from analysis import sentiment_analysis
from analysis.constants import (
    CUSTOM_STOPWORDS,
    FASTTEXT_MODEL_PATH,
    RANDOM_SEED,
    TOP_RESTAURANT_NATIONALITIES_IN_THE_NETHERLANDS,
    WORD2VEC_MODEL_PATH,
)
from analysis.data_import import (
    group_texts_by_decade,
    group_texts_by_issue,
    group_texts_by_page,
    group_texts_by_year,
    issues_starting_from_year,
)
from analysis.embeddings import (
    build_diachronic_embeddings_similarity_tables,
    build_embeddings_similarity_tables,
    get_most_similar_word_embedding_words,
    load_word2vec_model,
    plot_similarity_over_time,
    render_similarity_tables,
    train_fasttext_model,
    train_word2vec_model,
)
from analysis.preprocessing import (
    lemmatize_words,
    lowercase_texts,
    remove_punctuation,
    remove_stopwords,
    strip_unknown_words,
    word_count_dataset,
)
from analysis.topic_modelling import (
    compute_bertopic_coherence_score,
    compute_bertopic_topic_diversity,
    fit_lda_model,
    get_bertopic_topics_dict,
    print_lda_topics,
    lda_topics_to_dataframe,
    render_bertopic_table_latex,
    render_bertopic_topics_as_json,
)
from analysis.types import TextsByTimeSlice
from analysis.utils import aggregate_texts, get_most_common_words, time_function
from analysis.word_frequencies import (
    compute_term_frequencies_by_time_slice,
    plot_term_frequencies,
    plot_values_by_index,
    plot_word_counts_over_time,
    word_counts_per_time_slice,
    word_counts_per_issue,
)

from analysis.tfidf import (
    compute_tfidf,
    get_most_distinctive_words
)

print("Setting random seeds...")
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def word_frequencies(
    category_term_lists: dict[str, list[str]],
    texts_by_time_slice: TextsByTimeSlice,
    time_slice_name: str,
) -> dict[str, dict[int, float]]:
    print("Computing word frequencies...")

    start_time = time.time()
    term_frequencies_by_decade = compute_term_frequencies_by_time_slice(
        texts_by_time_slice,
        category_term_lists,
    )
    end_time = time.time()

    print(
        f"Computing term frequencies per {time_slice_name} took {end_time - start_time:.2f} seconds"
    )

    with open(
        f"{SCRIPT_DIR}/TEMP_term_frequencies_by_decade.json", "w", encoding="utf-8"
    ) as f:
        json.dump(term_frequencies_by_decade, f, ensure_ascii=False, indent=2)

    return term_frequencies_by_decade


if __name__ == "__main__":
    # -----Data Import-----
    print("Importing data...")
    SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
    DATA_PATH = os.path.join(SCRIPT_DIR, "allerhande_full_website_ocr.json")
    DUTCH_WORDS_PATH = os.path.join(SCRIPT_DIR, "dutch_words.txt")
    COUNTRY_LIST_PATH = os.path.join(SCRIPT_DIR, "country_list.json")

    issues: dict[str, dict[str, str]] = json.load(open(DATA_PATH))

    texts_by_issue = group_texts_by_issue(issues)
    texts_by_page = group_texts_by_page(issues)
    texts_by_year = group_texts_by_year(issues)
    texts_by_decade = group_texts_by_decade(issues)

    word_counts_per_year = word_counts_per_time_slice(texts_by_year)

    plot_word_counts_over_time(word_counts_per_year)

    counts_per_issue = word_counts_per_issue(texts_by_issue)

    plot_values_by_index(counts_per_issue)

    print("Importing country list...")
    dutch_country_adjectives_path = f"{SCRIPT_DIR}/dutch_country_adjectives.json"
    with open(dutch_country_adjectives_path, "r", encoding="utf-8") as f:
        country_adjectives: dict[str, list[str]] = json.load(f)

    netherlands_terms: list[str] = country_adjectives["nederland"] + [
        "nederland",
        "holland",
        "vaderlands",
        "vaderlandse",
    ]

    non_netherlands_terms: list[str] = []

    for country, adjectives in country_adjectives.items():
        if country != "nederland":
            non_netherlands_terms.extend(adjectives)
            non_netherlands_terms.extend([country])

    exotic_terms = ["exotisch", "exotische"]
    indonesian_terms = [
        "indonesisch",
        "indonesische",
        "indisch",
        "indische",
        "indie",
        "indische",
        "indonesië",
        "indonesiërs",
        "indonesiër",
    ]

    surinamese_terms = [
        "suriname",
        "surinaams",
        "surinaamse",
        "surinaamsche",
        "surinamer",
        "surinamers",
    ]

    caribbean_terms = [
        "antillen",
        "antilliaans",
        "antilliaanse",
        "aruba",
        "arubaans",
        "arubaanse",
        "bonaire",
        "bonairiaans",
        "bonairiaanse",
        "curaçao",
        "curaçaos",
        "curaçaose",
        "sint maarten",
        "sint maartens",
        "sint maartense",
        "saba",
        "sabaans",
        "sabaanse",
        "st. eustatius",
        "statiaans",
        "statiaanse",
    ]

    highlighted_country_names = TOP_RESTAURANT_NATIONALITIES_IN_THE_NETHERLANDS.copy()
    highlighted_country_adjectives: list[str] = []

    for country in highlighted_country_names:
        highlighted_country_adjectives.extend(country_adjectives[country])

    # Remove all values in highlighted_country_adjectives that end in 'e', to avoid using both 'Italiaanse' and 'Italiaans', etc.
    highlighted_country_adjectives = [
        word for word in highlighted_country_adjectives if not word.endswith("e")
    ]

    # Remove 'indisch' as adjective of 'Indonesisch' since it's not technically correct and has similar results
    highlighted_country_adjectives.remove("indisch")

    # -----Word frequencies-----
    print("Computing word frequencies...")
    dutch_vs_non_dutch_category_term_lists = {
        "Dutch": netherlands_terms,
        "non-Dutch": non_netherlands_terms,
    }

    word_frequencies(
        dutch_vs_non_dutch_category_term_lists,
        texts_by_time_slice=texts_by_decade,
        time_slice_name="decade",
    )

    former_colonies_term_lists = {
        "Surinamese": surinamese_terms,
        "Indonesian": indonesian_terms,
        "Caribbean": caribbean_terms,
        "Dutch": netherlands_terms,
    }

    start_time = time.time()
    freqs = word_frequencies(former_colonies_term_lists, texts_by_decade, "decade")
    end_time = time.time()
    print(f"Computing term frequencies took {end_time - start_time:.2f} seconds")

    plot_term_frequencies(
        freqs,
        title=f"Normalized mentions of country-related terms by decade",
        x_label="decade",
    )

    all_countries_term_lists = country_adjectives.copy()
    del all_countries_term_lists["nederland"]

    freqs = word_frequencies(
        all_countries_term_lists,
        texts_by_time_slice=texts_by_decade,
        time_slice_name="decade",
    )

    top_10_country_adjectives = {
        country: country_adjectives[country]
        for country in TOP_RESTAURANT_NATIONALITIES_IN_THE_NETHERLANDS
        if country in country_adjectives
    }
    top_10_country_adjectives["nederland"] = country_adjectives["nederland"]

    start_time = time.time()
    freqs = word_frequencies(top_10_country_adjectives, texts_by_decade, "decade")
    end_time = time.time()
    print(f"Computing term frequencies took {end_time - start_time:.2f} seconds")

    plot_term_frequencies(
        freqs,
        title=f"Normalized mentions of country-related terms by decade",
        x_label="decade",
    )

    ...

    # -----Preprocessing-----
    print("Preprocessing...")

    known_words = open(DUTCH_WORDS_PATH).read().splitlines()

    texts_by_issue_without_unknown_words = strip_unknown_words(texts_by_issue, known_words)
    texts_by_issue_without_unknown_words = [
        text.replace("ca", "") for text in texts_by_issue_without_unknown_words
    ]

    count1 = word_count_dataset(texts_by_issue)
    count2 = word_count_dataset(texts_by_issue_without_unknown_words)
    ratio = count2 / count1
    print(
        f"Word count before/after removing unknown words: {count1}/{count2} ({ratio:.2f})"
    )

    texts_by_page_without_unknown_words = strip_unknown_words(texts_by_page, known_words)
    texts_by_page_without_unknown_words = [
        text.replace("ca", "") for text in texts_by_page_without_unknown_words
    ]

    texts_cleaned = remove_punctuation(texts_by_page)
    texts_cleaned = lowercase_texts(texts_cleaned)
    texts_cleaned_unstopped = remove_stopwords(texts_cleaned, "dutch", CUSTOM_STOPWORDS)

    texts_cleaned_unstopped_lemmatized = lemmatize_words(texts_cleaned_unstopped)
    with open(f"{SCRIPT_DIR}/texts_by_page_cleaned_unstopped_lemmatized.txt", "w") as f:
        f.write("\n".join(texts_cleaned_unstopped_lemmatized))

    ...

    # -----BERTopic-----

    # These imports are here rather than at the top, because they take a long time and should be commented out when not needed
    from bertopic import BERTopic
    from umap import UMAP

    umap = UMAP(
        n_neighbors=15,
        n_components=5,
        min_dist=0.0,
        metric="cosine",
        low_memory=False,
        random_state=1337,
    )
    bertopic = BERTopic(language="dutch", umap_model=umap)
    topics, _ = time_function(bertopic.fit_transform, texts_cleaned_unstopped_lemmatized)
    topics_dict = get_bertopic_topics_dict(bertopic, topics)

    bertopic_model_path = (
        f"{SCRIPT_DIR}/models/BERTopic/bertopic_topics_by_page_with_preprocessing"
    )
    bertopic.save(bertopic_model_path)
    bertopic = BERTopic.load(bertopic_model_path)
    print("Computing diversity score...")
    diversity_score = compute_bertopic_topic_diversity(bertopic)
    print(f"diversity score: {diversity_score}")
    print("Computing coherence score...")

    coherence_score = compute_bertopic_coherence_score(bertopic, texts_by_page)
    print("Coherence:")
    print(coherence_score)

    topics = bertopic.get_topics()
    info = bertopic.get_document_info(docs=texts_cleaned_unstopped_lemmatized)

    bertopic_topics_json = render_bertopic_topics_as_json(
        bertopic_model=bertopic, topics=topics
    )

    json_path = f"{SCRIPT_DIR}/output/bertopic_topics_by_page_with_preprocessing.json"

    with open(json_path, "w", encoding="utf-8") as f:
        f.write(bertopic_topics_json)

    with open(json_path, "r", encoding="utf-8") as f:
        bertopic_topics = json.load(f)
    with open(json_path, "r", encoding="utf-8") as f:
        bertopic_topics_json = json.load(f)

    table = render_bertopic_table_latex(bertopic_topics_json)

    ...

    # -----TFIDF-----

    tfidf, result = time_function(compute_tfidf, texts_cleaned_unstopped)
    feature_names = tfidf.get_feature_names_out()
    most_distinctive_words = get_most_distinctive_words(result, feature_names)

    rs = random.sample(most_distinctive_words, 10)

    # -----FastText-----
    print("Training FastText model...")
    texts_cleaned_unstopped_lemmatized_lowercased = [text.lower() for text in texts_cleaned_unstopped_lemmatized]
    tokenized_texts = [text.split() for text in texts_cleaned_unstopped_lemmatized_lowercased]
    fasttext_model: FastText = time_function(train_fasttext_model, tokenized_texts)

    fasttext_model: FastText = FastText.load(FASTTEXT_MODEL_PATH)  # type: ignore

    most_similar_words = get_most_similar_word_embedding_words(
        fasttext_model, query_words=netherlands_terms + non_netherlands_terms
    )

    filtered_word_vectors = np.array(
        [fasttext_model.wv[word] for word in most_similar_words]
    )

    highlighted_country_names_fasttext_similarities = {
        k: most_similar_words[k] for k in highlighted_country_names
    }

    all_tables = render_similarity_tables(
        highlighted_country_names,
        highlighted_country_names_fasttext_similarities,
        table_title="FastText",
        query_words_per_table=3,
        similar_words_per_query_word=15
    )

    pyperclip.copy("\n".join(all_tables))

    # -----Word2Vec-----
    # print("Training Word2Vec model...")
    tokenized_texts = [text.split() for text in texts_cleaned_unstopped]
    word2vec_model = time_function(train_word2vec_model, tokenized_texts)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    WORD2VEC_MODEL_DIR = f"{SCRIPT_DIR}/models"
    model_path = f"{WORD2VEC_MODEL_DIR}/word2vec_model_{timestamp}.model"
    word2vec_model.save(model_path)

    models_per_decade: dict[int, Word2Vec] = {}
    query_words = list(top_10_country_adjectives.keys())

    for decade, issues in sorted(texts_by_decade.items()):
        decade_model_path = f"{WORD2VEC_MODEL_DIR}/word2vec_model_{decade}.model"
        decade_word2vec_model = Word2Vec.load(decade_model_path)

        models_per_decade[decade] = decade_word2vec_model

    # Plot cosine similarities over time for a target word and related words
    plot_similarity_over_time(
        model_dir=WORD2VEC_MODEL_DIR,
        decades=sorted(texts_by_decade.keys()),
        target_word="indonesisch",
        related_words=["koreaans", "thais", "surinaams"],
    )

    texts = aggregate_texts(issues)
    texts_cleaned = remove_punctuation(texts)
    texts_cleaned = lowercase_texts(texts_cleaned)
    texts_cleaned_unstopped = remove_stopwords(
        texts_cleaned, "dutch", CUSTOM_STOPWORDS
    )

    tokenized_texts = [text.split() for text in texts_cleaned_unstopped]

    print(f"Training Word2Vec model for decade {decade}...")
    decade_word2vec_model = time_function(train_word2vec_model, tokenized_texts)
    decade_word2vec_model.save(decade_model_path)

    diachronic_similarity_tables = build_diachronic_embeddings_similarity_tables(
        models_per_decade,
        query_words,
        table_title="Word associations by decade",
        topn=3,
    )

    for table in diachronic_similarity_tables:
        pyperclip.copy(table)
        ...
