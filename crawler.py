from bs4 import BeautifulSoup
import json
import os
import re
import requests
import time

BASE_URL = "https://albertheijnerfgoed.courant.nu"


def get_all_issue_links() -> list[str]:
    """
    Crawl the site to get the links to all of the issues of Allerhande
    """
    with open("issue-links.txt", "r") as f:
        issue_links = f.read().splitlines()

    all_magazine_links = []

    for issue_link in issue_links:
        response = requests.get(issue_link)

        soup = BeautifulSoup(response.content, "html.parser")
        # grid of all the magazines published in this year
        issue_overview_div = soup.select_one("div.row.cv")
        if issue_overview_div:
            link_tags = issue_overview_div.select("a[href]")
            for link_tag in link_tags:
                href_value = link_tag.get("href")
                print(href_value)
                all_magazine_links.append(href_value)

    return all_magazine_links


def get_all_issue_pages_links(magazine_links: list[str]) -> dict[str, list[str]]:
    """
    For each issue of Allerhande, get the links to all of the pages
    """
    magazine_page_urls = {}

    for i, magazine_link in enumerate(magazine_links, 1):
        magazine_page_urls[magazine_link] = []
        page_number = 1

        while True:
            page_link = re.sub(r"(page/)\d+", f"page/{page_number}", magazine_link)
            response = requests.head(page_link)
            if response.status_code == 404:
                break
            magazine_page_urls[magazine_link].append(page_link)

            page_number += 1
        print(f"Done {i} out of 707 links")

    return magazine_page_urls

PageNumber = str
Text = str
IssueDate = str


def get_ocr_texts(
    issue_page_links: dict[str, list[str]],
) -> dict[IssueDate, dict[PageNumber, Text]]:
    total_pages = sum(len(page_links) for page_links in issue_page_links.values())
    ocr_texts: dict[IssueDate, dict[PageNumber, Text]] = {}
    pages_processed = 0  # Track the number of processed pages

    for issue_date, page_links in issue_page_links.items():
        if issue_date not in ocr_texts:
            ocr_texts[issue_date] = {}

        for i, page_link in enumerate(page_links):
            response = requests.get(page_link)
            soup = BeautifulSoup(response.content, "html.parser")
            tag = soup.select_one("span#ocr-text")

            if not tag:
                continue

            ocr_texts[issue_date][str(i)] = tag.text
            pages_processed += 1
            print(f"Processed {pages_processed}/{total_pages} pages")
    return ocr_texts


def save_ocr_texts_to_file(issues_pages_file_path: str, output_file_path: str) -> None:
    with open(issues_pages_file_path, "r") as f:
        issues_pages = json.load(f)
        ocr_texts = get_ocr_texts(issues_pages)
        with open(output_file_path, "w") as f:
            f.write(json.dumps(ocr_texts))


if __name__ == "__main__":
    SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
    JSON_FILE_PATH = os.path.join(SCRIPT_DIR, "magazine_page_ranges.json")
    OCR_OUTPUT_FILE_PATH = os.path.join(SCRIPT_DIR, "allerhande_full_website_ocr.json")
    save_ocr_texts_to_file(JSON_FILE_PATH, OCR_OUTPUT_FILE_PATH)
