from typing import TypedDict
from sklearn.pipeline import Pipeline


class SentimentResult(TypedDict):
    text: str
    label: str
    score: float


def sentiment_analysis(
    analysis_pipeline: Pipeline, texts: list[str]
) -> list[SentimentResult]:
    results = []
    skipped_counter = 0

    for i, text in enumerate(texts, start=1):
        print(f"Analyzing text #{i}, skipped: {skipped_counter}", end="\r")
        # Skip texts that are too long for the model
        if len(text) > 512:
            skipped_counter += 1
            continue
        result = analysis_pipeline(text)[0]  # type: ignore
        results.append(
            {"text": text, "label": result["label"], "score": result["score"]}  # type: ignore
        )
    return results
