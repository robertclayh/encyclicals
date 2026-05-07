# Text as Data — Final Project Guide

**Clay Harris (jbm2rt@virginia.edu) / Text as Data / 2026-05-07**

---

## 1. Overview

The final project applies the complete **Standard Text Analytic Data Model (STADM)** pipeline to a substantial corpus of your choosing. You will collect, process, annotate, model, and analyze a body of text, demonstrating mastery of the methods covered in the course.

The project is worth **30% of the final grade** and is due at the end of the semester. Submissions are individual (no group projects).

> **This corpus:** Papal Encyclicals and Church Council Documents from the Vatican archives, spanning roughly 325 AD to the present — approximately 700+ documents covering the full history of Catholic teaching.

---

## 2. Deliverables

All deliverables should be submitted via the course portal. The core deliverables are:

### 2.1 Required Data Tables (CSV files in `data/processed/`)

| File | Description |
|------|-------------|
| `LIBRARY.csv` | Document-level metadata indexed by `doc_id` |
| `TOKEN.csv` | Token table with OHCO index `(doc_id, para_num, sent_num, token_num)` |
| `VOCAB.csv` | Vocabulary table indexed by `term_str` |
| `TFIDF_DTM.csv` | Document-term matrix with TFIDF weights |
| `DOC_PCA.csv` | Document principal components (rows = docs, cols = PC0…PCn) |
| `LOADINGS.csv` | Term loadings on each principal component |
| `explained_variance.csv` | Variance explained per PC |
| `DOC_TOPICS.csv` | Document-topic distributions from LDA |
| `TOPIC_TERMS.csv` | Topic-term weight matrix |
| `EMBEDDINGS.csv` | Word2Vec term embeddings |
| `DOC_SENTIMENT.csv` | Document-level VADER sentiment scores |

### 2.2 Required Notebooks

| Notebook | Description |
|----------|-------------|
| `notebooks/01_scraping.ipynb` | Corpus acquisition and language filtering |
| `notebooks/02_pipeline.ipynb` | Full F0→F5 pipeline execution with statistics |

### 2.3 Source Modules

| File | Description |
|------|-------------|
| `src/scraper.py` | Web scraping, language detection, dead-link handling |
| `src/pipeline.py` | All pipeline functions F1–F5 |

### 2.4 Final Report

A written report (see Section 7) that interprets the analytical results. The report should be a separate document (PDF or Markdown) accompanying the notebooks.

---

## 3. Corpus Requirements

- Minimum **2,000 documents** (or paragraphs serving as the document unit).
- Text must be **free-form prose** (not structured/tabular data).
- Must be collected programmatically (scraping, API, or archive download) — document the process.
- The **document unit** should reflect a meaningful discursive boundary (a document, chapter, paragraph, etc.).

### 3.1 F0 Source Data

Raw text files live in `data/raw/`. Each file is named by `doc_id` (snake-case, matching `encyclicals_index.json`). The metadata index (`data/encyclicals_index.json`) records:

```json
{
  "doc_id": "...",
  "pope": "...",
  "title": "...",
  "year": "...",
  "language": "en",
  "url": "..."
}
```

---

## 4. Required Tables — Column Specifications

### 4.1 LIBRARY.csv

Index: `doc_id`

| Column | Type | Description |
|--------|------|-------------|
| `pope` | str | Author / pope name |
| `title` | str | Document title |
| `year` | int/str | Year of publication |
| `n_paragraphs` | int | Number of paragraphs |
| `n_chars` | int | Character count |
| `n_tokens` | int | Token count |
| `sentiment_neg` | float | VADER negative score |
| `sentiment_neu` | float | VADER neutral score |
| `sentiment_pos` | float | VADER positive score |
| `sentiment_compound` | float | VADER compound score |

### 4.2 TOKEN.csv

Index: `(doc_id, para_num, sent_num, token_num)` — the **OHCO index**

| Column | Type | Description |
|--------|------|-------------|
| `token_str` | str | Raw token (original casing) |
| `term_str` | str | Lowercased token |
| `pos` | str | Penn Treebank POS tag |
| `lemma` | str | WordNet lemma |
| `is_stop` | bool | Stopword flag |
| `is_alpha` | bool | Alphabetic flag |
| `tfidf` | float | TFIDF weight for this token in its document |

### 4.3 VOCAB.csv

Index: `term_str`

| Column | Type | Description |
|--------|------|-------------|
| `n` | int | Total token count |
| `df` | int | Document frequency |
| `idf` | float | Inverse document frequency |
| `pos` | str | Most common POS tag |
| `lemma` | str | Most common lemma |
| `is_stop` | bool | Stopword flag |
| `vader_neg` | float | VADER negative score |
| `vader_neu` | float | VADER neutral score |
| `vader_pos` | float | VADER positive score |
| `vader_compound` | float | VADER compound score |
| `max_tfidf` | float | Maximum TFIDF across all documents |

### 4.4 TFIDF_DTM.csv

- Rows: `doc_id`
- Columns: `term_str` (filtered: alphabetic, non-stop, `df > 1`)
- Values: normalized TFIDF weight

### 4.5 DOC_SENTIMENT.csv

Index: `doc_id`

| Column | Type | Description |
|--------|------|-------------|
| `vader_compound_mean` | float | Mean compound score over doc tokens |
| `vader_pos_mean` | float | Mean positive score |
| `vader_neg_mean` | float | Mean negative score |
| `vader_neu_mean` | float | Mean neutral score |

---

## 5. Pipeline Stages (F0 → F5)

### F0: Raw Corpus (Source Text)
The original downloaded `.txt` files in `data/raw/`. Acquired by `src/scraper.py` and driven by `notebooks/01_scraping.ipynb`.

### F1: Machine Learning Corpus Format
Convert raw text to a structured paragraph table. Each row represents one paragraph with doc metadata.

**Key function:** `build_f1_corpus(english_only=True)`

### F2: STADM Tables
Tokenize F1 paragraphs and produce:
- **LIBRARY** — document-level metadata
- **TOKEN** — every token with OHCO positional index
- **VOCAB** — unique terms with frequency statistics

**Key function:** `build_f2_tables(corpus)`

### F3: NLP Annotations
Enrich TOKEN and VOCAB with:
- Part-of-speech (POS) tags via NLTK averaged perceptron tagger
- Lemmas via NLTK WordNetLemmatizer
- Stopword flags
- VADER sentiment scores (token-level → aggregated to VOCAB and doc-level)

**Key function:** `build_f3_annotations(TOKEN, VOCAB, LIBRARY)`

**New output:** `DOC_SENTIMENT.csv` — per-document aggregate VADER scores

### F4: TFIDF Vectorization
Compute term frequency (TF), IDF, and TFIDF. Build document-term matrix.

**Key function:** `build_f4_tfidf(TOKEN, VOCAB, LIBRARY)`

### F5: Unsupervised Models
Fit three models:
1. **PCA** — dimensionality reduction on TFIDF matrix → `DOC_PCA`, `LOADINGS`, `explained_variance`
2. **LDA** — topic modeling → `DOC_TOPICS`, `TOPIC_TERMS`
3. **Word2Vec** — term embeddings → `EMBEDDINGS`

**Key function:** `build_f5_models(LIBRARY, TOKEN, VOCAB, TFIDF_DTM, ...)`

---

## 6. Format and Style Requirements

All notebooks and source files must include the following **Text as Data header** as the first cell (Markdown) or module docstring:

```
Name (email@domain)
Text as Data
Date
```

**Example (notebook cell):**
```markdown
# [Notebook Title]
**Clay Harris (jbm2rt@virginia.edu) / Text as Data / 2026-05-07**
```

### 6.1 Notebook Structure Requirements

Each pipeline stage section must contain:
1. A **Markdown header** (`## F1: ...`) with an explanatory paragraph describing what the stage does
2. A `%%time` magic cell for the compute step
3. A **statistics cell** reporting counts (e.g., `print(f"TOKEN: {len(TOKEN):,}")`)
4. A `.head()` display of the resulting DataFrame

### 6.2 Config Block

Every notebook should have a **configuration cell near the top** with all tuneable parameters:

```python
# ── Configuration ──────────────────────────────────────────────
ENGLISH_ONLY  = True
N_COMPONENTS  = 10
N_TOPICS      = 10
W2V_DIM       = 100
FORCE_RERUN   = False   # Set True to recompute all stages
```

### 6.3 Code Quality

- Use docstrings for all functions
- Use `logging` (not bare `print`) in `src/*.py` modules
- Notebooks may use `print()` for student-facing output
- Import all libraries at the top of the notebook
- No hard-coded paths — use `pathlib.Path` constants

---

## 7. Final Report Requirements

The written report (40% of grade) must include the following sections:

### 7.1 Introduction
- Describe the corpus: what it is, why you chose it, what questions you hope to answer
- Describe the collection process (scraper, source URL, date collected)

### 7.2 Data Description
- Corpus statistics: number of documents, tokens, vocabulary size
- Distribution of documents over time, pope, or other metadata dimension
- Example documents (show a few representative items from LIBRARY)

### 7.3 Exploratory Analysis
- Token frequency distributions (Zipf's law)
- Top terms by TFIDF
- VADER sentiment distribution across documents and over time

### 7.4 Topic Modeling (LDA)
- Describe each topic using its top terms
- Show document-topic assignments
- Interpret the topics in the context of the corpus

### 7.5 Principal Component Analysis (PCA)
- Explained variance curve (scree plot)
- Interpretation of top PCs using term loadings
- Visualization of documents in PC space

### 7.6 Word Embeddings (Word2Vec)
- Most similar terms to a few key words
- 2D projection (t-SNE or PCA) of embedding space
- Interpretation of semantic clusters

### 7.7 Conclusion
- Summary of findings
- Limitations
- Future directions

---

## 8. Grading Rubric

| Category | Weight | Description |
|----------|--------|-------------|
| **Deliverables** | 50% | All required CSV files present and correctly structured; notebooks run end-to-end |
| **Legibility / Format** | 10% | Text as Data headers, `%%time` cells, stats output, `.head()` displays, clean Markdown |
| **Final Report** | 40% | Quality of analysis, interpretation, and writing across all 7 report sections |

### 8.1 Deliverables Rubric Detail (50%)

| Item | Points | Criteria |
|------|--------|----------|
| LIBRARY.csv | 5 | Present, correct index, all required columns |
| TOKEN.csv | 10 | OHCO index, all annotation columns, correct values |
| VOCAB.csv | 5 | Present, all sentiment/POS/lemma columns |
| TFIDF_DTM.csv | 5 | Present, correct shape, numeric values |
| DOC_PCA / LOADINGS / explained_variance | 5 | Present, interpretable |
| DOC_TOPICS / TOPIC_TERMS | 5 | Present, topics make sense |
| EMBEDDINGS.csv | 5 | Present, correct dimensions |
| DOC_SENTIMENT.csv | 5 | Present, doc-level VADER scores |
| Notebooks execute | 5 | Both notebooks run top-to-bottom without errors |

### 8.2 Legibility Rubric Detail (10%)

- Text as Data header in every notebook
- Section headers with explanatory text
- `%%time` on all expensive cells
- Stats printed after each stage
- `.head()` shown for each table

### 8.3 Final Report Rubric Detail (40%)

- Introduction (5%)
- Data Description (5%)
- Exploratory Analysis (10%)
- Topic Modeling (5%)
- PCA (5%)
- Word Embeddings (5%)
- Conclusion (5%)

---

## 9. Deliverables Checklist

Use this checklist to verify your submission is complete:

### Data Files
- [ ] `data/processed/LIBRARY.csv`
- [ ] `data/processed/TOKEN.csv`
- [ ] `data/processed/VOCAB.csv`
- [ ] `data/processed/TFIDF_DTM.csv`
- [ ] `data/processed/DOC_PCA.csv`
- [ ] `data/processed/LOADINGS.csv`
- [ ] `data/processed/explained_variance.csv`
- [ ] `data/processed/DOC_TOPICS.csv`
- [ ] `data/processed/TOPIC_TERMS.csv`
- [ ] `data/processed/EMBEDDINGS.csv`
- [ ] `data/processed/DOC_SENTIMENT.csv`

### Notebooks
- [ ] `notebooks/01_scraping.ipynb` — Text as Data header present, executes clean
- [ ] `notebooks/02_pipeline.ipynb` — Text as Data header present, all F1–F5 stages, stats, `.head()`, executes clean

### Source Code
- [ ] `src/scraper.py` — documented, checkpoint-aware
- [ ] `src/pipeline.py` — all F1–F5 functions, `--resume` and `--force-stage` CLI flags, DOC_SENTIMENT output

### Report
- [ ] Final report PDF or Markdown with all 7 sections

---

## 10. Repository Structure

```
encyclicals/
├── README.md
├── FINAL_PROJECT_GUIDE.md          ← this file
├── requirements.txt
├── data/
│   ├── encyclicals_index.json      ← document metadata index
│   ├── raw/                        ← F0 source .txt files
│   └── processed/                  ← all output CSV files
│       └── checkpoints/            ← pickle caches for resume
├── notebooks/
│   ├── 01_scraping.ipynb
│   └── 02_pipeline.ipynb
├── src/
│   ├── __init__.py
│   ├── scraper.py
│   └── pipeline.py
└── output/                         ← visualizations and report figures
```

---

## 11. Environment Setup

```bash
# Create virtual environment
python -m venv .venv
.venv\Scripts\activate      # Windows
source .venv/bin/activate   # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Download NLTK data (first run)
python -c "
import nltk
for pkg in ['punkt_tab', 'averaged_perceptron_tagger_eng', 'wordnet',
            'stopwords', 'vader_lexicon', 'omw-1.4']:
    nltk.download(pkg, quiet=True)
"
```

### requirements.txt (key packages)

```
nltk
scikit-learn
gensim
pandas
numpy
tqdm
langdetect
requests
beautifulsoup4
```

---

## 12. Running the Pipeline

### Option A: Notebook (recommended for interactive exploration)

Open `notebooks/02_pipeline.ipynb`. Set `FORCE_RERUN = False` to use cached stages (fast), or `FORCE_RERUN = True` to recompute from scratch.

### Option B: Command line

```bash
# Full pipeline
python src/pipeline.py

# Resume from last checkpoint
python src/pipeline.py --resume

# Force recompute from F3 onward (preserves F1/F2 caches)
python src/pipeline.py --force-stage f3

# Tune model parameters
python src/pipeline.py --n-components 15 --n-topics 15 --w2v-dim 200

# English-only documents
python src/pipeline.py --english-only
```

---

*This guide was generated from the course rubric and project requirements for Text as Data.*
