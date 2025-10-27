import os
import re

from analysis.types import TextsByTimeSlice, TextsByYear


def list_files_recursive(base_path: str) -> list[str]:
    all_files = []
    for dirpath, _, filenames in os.walk(base_path):
        for filename in filenames:
            all_files.append(os.path.join(dirpath, filename))
    return all_files


# returns dict[IssueDate, dict[PageNumber, PageText]]
def import_text_data(base_path: str, error_path: str) -> dict[str, dict[str, str]]:
    """
    Import the OCR data from the specified base path.

    Args:
        base_path (str): The base path to the OCR data.

    Returns:
        dict[str, list[str]]: The OCR data as a dictionary of page numbers and lists of words.
    """
    files = list_files_recursive(base_path)
    txt_files = [f for f in files if f.endswith(".txt")]

    all_data = {}
    for file_path in txt_files:
        with open(file_path) as file:
            issue_date = get_issue_date_from_path(file_path)
            page_number = file.name.split("/")[-1].split(".")[0]
            try:
                page_text = file.read()
            except UnicodeDecodeError as e:
                with open(error_path, "a") as error_file:
                    error_file.write(f"UnicodeDecodeError reading {file_path}: {e}\n")
                continue

            if issue_date not in all_data:
                all_data[issue_date] = {}

            all_data[issue_date][page_number] = page_text
    return all_data


def issues_starting_from_year(
    data: dict[str, dict[str, str]], year: int
) -> dict[str, dict[str, str]]:
    issues = {}

    for issue_date, pages in data.items():
        if int(issue_date.split("-")[0]) >= year:
            issues[issue_date] = pages
    return issues


def group_texts_by_issue(issues: dict[str, dict[str, str]]) -> list[str]:
    issue_texts: list[str] = []

    for issue_date, pages in issues.items():
        issue_text = " ".join(pages.values())
        issue_texts.append(issue_text)

    return issue_texts


def group_texts_by_page(issues):
    page_texts = []

    for issue_date, pages in issues.items():
        for page_number, page_text in pages.items():
            page_texts.append(page_text)
    return page_texts


def group_texts_by_year(issues: dict[str, dict[str, str]]) -> TextsByYear:
    """
    Groups OCR page texts from issues by year.

    Args:
        issues: A dictionary mapping issue dates (YYYY-MM-DD) to page texts.
                Each page text is itself a dictionary mapping page numbers (str)
                to OCR-extracted text (str).

    Returns:
        A dictionary mapping years (e.g., 1950, 1951) to dictionaries of issues.
        Each list contains the page texts of all issues published in that decade.
    """

    year_texts: TextsByYear = {}

    for issue_date, pages in issues.items():
        year: int = parse_year_from_issue_date(issue_date)

        if not year in year_texts:
            year_texts[year] = {}
        year_texts[year][issue_date] = pages

    return year_texts


def group_texts_by_decade(issues: dict[str, dict[str, str]]) -> TextsByTimeSlice:
    """
    Groups OCR page texts from issues by decade.

    Args:
        issues: A dictionary mapping issue dates (YYYY-MM-DD) to page texts.
                Each page text is itself a dictionary mapping page numbers (str)
                to OCR-extracted text (str).

    Returns:
        A dictionary mapping decades (e.g., 1950, 1960) to dictionaries of issues.
        Each list contains the page texts of all issues published in that decade.
    """

    decade_texts: TextsByTimeSlice = {}

    for issue_date, pages in issues.items():
        decade: int = parse_decade_from_issue_date(issue_date)

        if not decade in decade_texts:
            decade_texts[decade] = {}
        decade_texts[decade][issue_date] = pages

    return decade_texts


def get_issue_date_from_path(path: str) -> str:
    match = re.search(r"(\d{4}-\d{2}-\d{2})", path)

    # If a match is found, extract the date
    if match:
        date = match.group(1)
        return date
    else:
        raise ValueError(f"No issue date found in path: {path}")


def parse_decade_from_issue_date(issue_date: str) -> int:
    year = int(issue_date[:4])
    return (year // 10) * 10


def parse_year_from_issue_date(issue_date: str) -> int:
    year = int(issue_date[:4])
    return year
