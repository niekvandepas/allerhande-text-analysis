# "Get a Taste of Asia"  
**Computationally Tracing the Culinary Other in Dutch Food Discourse**  
*Niek van de Pas · Ayoub Bagheri · Willy Sier*  
Part of the paper:  
> van de Pas, N., Bagheri, A., & Sier, W. (2025). "Get a Taste of Asia": Computationally Tracing the Culinary Other in Dutch Food Discourse.  

---

## Overview

This repository contains the analysis code accompanying the paper *“Get a Taste of Asia”*, which combines **anthropological theory** with **computational text analysis** to study how *Allerhande* — the Netherlands’ most widely read food magazine — represents Dutch and foreign cuisines from **1954 to 2025**.

The scripts implement a mixed-methods NLP pipeline integrating:
- **Frequency analysis**
- **Topic modeling**
- **Word embeddings** (FastText and Word2Vec)
- **Exploratory text statistics**

The pipeline allows tracing how culinary boundaries are constructed, how they evolve over time, and how Dutch cuisine is positioned relative to foreign culinary others.

---

## Setup

The analysis was developed and tested on:

```bash
Python 3.10.14
```

## Steps

1. **Clone the repository**

```bash
git clone <repository-url>
cd <repository-folder>
```

2.	Install dependencies

Make sure you are in a virtual environment. Then install all Python dependencies:

```python
pip install -r requirements.txt
```

3.	Download the data

Due to copyright restrictions, the full Allerhande content cannot be redistributed. You need to run the crawler script to generate the dataset:

```bash
python crawler.py
```

This will create the file `allerhande_full_website_ocr.json` locally.

4.	Set constants

```python
DATA_IMPORT_ERRORS_FILE_PATH = ""
FASTTEXT_MODEL_PATH = ""
WORD2VEC_MODEL_PATH = ""
```

5. Run the code

Run `allerhande_text_analysis.py` to run the code.
