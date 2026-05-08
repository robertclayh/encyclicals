# Papal Documents — Text as Data

DS 5001 — Text as Data Final Project

A digital critical edition of Catholic papal documents.

## Corpus Overview

**Documentation:** See [MANIFEST.md](MANIFEST.md) for detailed provenance, source descriptions, and data format specifications.

**Corpus:** Two primary source collections used in the final paper:
- Vatican website (modern popes; broad and deep coverage): https://www.vatican.va/content/vatican/en.html
- Papal Encyclicals Online (longer span; sparser documentation in places): https://www.papalencyclicals.net/document-directory

The collection hosts papal encyclicals, addresses, and related church documents from 1215 CE to 2026 CE, with particular focus on the modern era (1966-present), primarily in English translation.

### Key Statistics

- **Total Source Files (`data/raw/*.txt`):** 17,466
- **Total Documents (`data/processed/LIBRARY.csv`):** 17,352
- **Total Tokens (`data/processed/TOKEN.csv.gz`):** 28,425,031
- **Vocabulary Entries (`data/processed/VOCAB.csv`):** 219,533
- **Time Range (`LIBRARY.year`):** 1215 CE – 2026 CE
- **Modern Era (focus, 1966+):** 16,284 documents

## Project Structure

```
encyclicals/
├── src/
│   ├── scraper.py         # Web scraper for papalencyclicals.net
│   └── pipeline.py        # F0→F5 text analytics pipeline
├── notebooks/
│   ├── 01_scraping.ipynb   # Corpus acquisition
│   ├── 02_pipeline.ipynb   # Run the processing pipeline
│   └── 03_exploration.ipynb # Analysis and visualization
├── data/
│   ├── raw/               # Raw text files (one per encyclical)
│   └── processed/         # CSV tables (LIBRARY, TOKEN, VOCAB, etc.)
├── requirements.txt
└── README.md
```

## Pipeline Stages

| Stage | Description | Output |
|-------|-------------|--------|
| F0 | Raw scraped text (HTML/epub) | `data/raw/*.txt` |
| F1 | Paragraphs indexed by doc hierarchy | `F1_CORPUS.csv` |
| F2 | STADM: LIBRARY, TOKEN, VOCAB | Core tables |
| F3 | NLP: POS, lemma, stopwords, sentiment | Annotated tables |
| F4 | TFIDF vectorization | `TFIDF_DTM.csv` |
| F5 | PCA, LDA, word2vec | Model output tables |

## Quick Start

```bash
pip install -r requirements.txt

# Step 1: Scrape the corpus
python src/scraper.py

# Step 2: Run the full pipeline
python src/pipeline.py

# Step 3: Explore in Jupyter
jupyter notebook notebooks/03_exploration.ipynb
```

## Deliverables

### Data Files

Processed corpus data in STADM and analytical formats:

- **LIBRARY.csv** – Document metadata (pope, date, type, source)
- **TOKEN.csv** – Annotated word tokens with linguistic features
- **VOCAB.csv** – Vocabulary statistics and features
- **DOC_TOPICS.csv** – LDA topic model concentrations (10 topics)
- **TOPIC_TERMS.csv** – Top terms per topic with weights
- **DOC_PCA.csv** – Principal component analysis scores (10 components)
- **EMBEDDINGS.csv** – Word2Vec embeddings (100-dimensional)
- **DOC_SENTIMENT.csv** – Sentiment analysis (VADER compound scores)

See [MANIFEST.md](MANIFEST.md) for complete descriptions of all data tables.

### Final Report

See [final_report/DS5001_RCHarris_finalreport.pdf](final_report/DS5001_RCHarris_finalreport.pdf) for the full 2–4 page final report with research findings and visualizations.

**Key Research Questions:**

- **RQ1:** Do modern papacies have distinct topical and stylistic signatures?
- **RQ2:** How do encyclicals and speeches differ in topical focus?
- **RQ3:** Has Catholic discourse undergone measurable lexical-semantic drift over time?

### Notebooks

- **01_scraping.ipynb** – Web scraping pipeline for corpus acquisition
- **02_pipeline.ipynb** – F0→F5 text analytics processing pipeline
- **03_exploration.ipynb** – Exploratory data analysis and visualization

## Submission and Archive

This repository contains all code, notebooks, configuration files, and processed data tables. 

**Source files location:** Raw source files are hosted directly in this repository under `data/raw/`.

**Processed data:** All STADM and F5 analytical tables are included in the `data/processed/` directory and available in this repository.

**Final report:** The final report (IEEE conference format) is located at [final_report/DS5001_RCHarris_finalreport.pdf](final_report/DS5001_RCHarris_finalreport.pdf).
