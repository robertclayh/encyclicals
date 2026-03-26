# Papal Encyclicals — Exploratory Text Analytics

DS 5001 — Exploratory Text Analytics Final Project

A digital critical edition of Catholic papal encyclicals, tracing how Catholic
social teaching evolved over time through computational text analysis.

## Source

**Corpus:** Papal encyclicals from https://www.papalencyclicals.net/document-directory

The site hosts encyclicals from the 1700s to the present, organized by pope.
Documents are available as HTML text or epub files, primarily in English.

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

## Visualizations

The exploration notebook includes:
- Hierarchical cluster diagrams
- Heatmaps (topic distributions, term correlations)
- PCA scatter plots
- KDE plots (sentiment distributions)
- Dispersion plots (term usage across corpus)
- t-SNE plots (word embeddings)
