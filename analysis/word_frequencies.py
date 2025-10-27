import re
from typing import Dict, List
import matplotlib.pyplot as plt

from analysis.types import TextsByTimeSlice
from analysis.utils import aggregate_texts

Issues = Dict[str, Dict[str, str]]  # issue_date -> page_number -> text


def count_term_mentions_in_text(text: str, terms: List[str]) -> int:
    """
    Count how many times any of the terms appear in the given text.

    Args:
        text: A string to search.
        terms: List of lowercase terms (adjectives) to count.

    Returns:
        Total number of term occurrences in the text.
    """
    pattern = re.compile(
        r"\b(" + "|".join(re.escape(term) for term in terms) + r")\b", re.IGNORECASE
    )
    matches = pattern.findall(text)
    return len(matches)


def compute_term_frequencies_by_time_slice(
    texts_by_time_slice: TextsByTimeSlice, term_lists: dict[str, list[str]]
) -> Dict[str, Dict[int, float]]:
    """
    Compute normalized frequency of terms per time slice.

    Args:
        texts_by_time_slice: Dict mapping time slice (int indicating the year or decade (e.g., 1950)) to issues.
        term_lists: Dict mapping between categories (e.g., 'dutch', 'indian') to lists of corresponding terms

    Returns:
        Dictionary with normalized frequencies for each category per decade.
    """

    normalized_freqs = {}

    for category in term_lists.keys():
        normalized_freqs[category] = {}

    for decade, issues in sorted(texts_by_time_slice.items()):
        texts = aggregate_texts(issues)
        total_words = sum(len(text.split()) for text in texts)
        if total_words < 1:
            # Avoid division by zero if no words are found
            continue

        for category, category_terms in term_lists.items():
            category_count = sum(
                count_term_mentions_in_text(text, category_terms) for text in texts
            )

            normalized_freqs[category][decade] = category_count / total_words

    return normalized_freqs


from typing import Dict


def plot_term_frequencies(
    freqs: Dict[str, Dict[int, float]], title: str, x_label: str
) -> None:
    """
    Plot term frequencies per decade for any number of categories.

    Args:
        freqs: Dict mapping categories (e.g., 'dutch', 'foreign') to decade->frequency mappings.
        title: Plot title.
        x_label: X-axis label.
    """
    decades = sorted({decade for category in freqs.values() for decade in category})

    plt.figure(figsize=(10, 6))

    for category, decade_freqs in freqs.items():
        values = [decade_freqs.get(decade, 0) for decade in decades]
        plt.plot(decades, values, marker="o", label=category.capitalize())

    plt.xlabel(x_label)
    plt.ylabel("Mentions (normalized)")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def word_counts_per_time_slice(texts_by_time_slice: TextsByTimeSlice) -> Dict[int, int]:
    """
    Count total words per time slice.

    Args:
        texts_by_time_slice: Dict mapping time slice to issues.

    Returns:
        Dictionary mapping time slice to total word count.
    """
    word_counts = {}

    for time_slice, issues in sorted(texts_by_time_slice.items()):
        total_word_count = 0
        texts = aggregate_texts(issues)
        for text in texts:
            total_word_count += len(text.split())

        word_counts[time_slice] = total_word_count
        print(f"Time slice {time_slice}: {total_word_count} words")

    return word_counts

def word_counts_per_issue(texts_by_issue: list[str]) -> list[int]:
    """
    Count total words per issue.

    Args:
        texts_by_issue: list of texts

    Returns:
        List of word counts
    """
    word_counts = []

    for text in texts_by_issue:
        total_word_count = len(text.split())

        word_counts.append(total_word_count)

    return word_counts

from typing import Dict

def plot_word_counts_over_time(
    word_counts: Dict[int, int],
    title: str = "Word Counts Over Time",
    x_label: str = "Year",
    y_label: str = "Word Count"
) -> None:
    """
    Plot total word counts per year.

    Args:
        word_counts: Dict mapping year -> word count.
        title: Plot title.
        x_label: Label for x-axis.
        y_label: Label for y-axis.
    """
    years = sorted(word_counts.keys())
    counts = [word_counts[year] for year in years]

    plt.figure(figsize=(12, 6))
    plt.plot(years, counts, marker="o")
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_values_by_index(
    values: List[int],
    title: str = "Values by Index",
    x_label: str = "Index",
    y_label: str = "Value"
) -> None:
    """
    Plot a list of values with their list index on the x-axis.

    Args:
        values: List of numerical values.
        title: Plot title.
        x_label: Label for x-axis.
        y_label: Label for y-axis.
    """
    indices = list(range(len(values)))

    plt.figure(figsize=(12, 6))
    plt.plot(indices, values, marker="o", linestyle="-", color="tab:blue")
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()
