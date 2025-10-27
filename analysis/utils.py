import time
from collections import Counter
import re

from analysis.types import Issues


def time_function(func, *args, **kwargs):
    start_time = time.time()
    result = func(*args, **kwargs)
    print(f"Function '{func.__name__}' took {time.time() - start_time:.2f} seconds")
    return result


def get_most_common_words(texts: list[str]) -> Counter:
    word_counts = Counter()

    for text in texts:
        words = re.findall(
            r"\b\w+\b", text.lower()
        )  # Extract words (normalize to lowercase)
        word_counts.update(words)  # Count occurrences

    return word_counts


def aggregate_texts(issues: Issues) -> list[str]:
    """
    Aggregate all page texts of issues into a list of texts.

    Args:
        issues: Dict mapping issue_date to dict of page_number to text.

    Returns:
        List of page texts (str).
    """
    texts = []
    for issue_pages in issues.values():
        texts.extend(issue_pages.values())
    return texts
