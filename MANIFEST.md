# Corpus Manifest: Papal and Conciliar Documents

**Author:** Clay Harris (jbm2rt@virginia.edu)  
**Course:** DS 5001 - Text as Data
**Date:** May 2026

## Provenance

### Sources

This corpus comprises two primary collections used in the final paper:

1. **Vatican Website Collection (modern popes; broad and deep coverage)**
  - **Source:** The Holy See website
  - **Description:** Official Vatican-hosted texts for modern papacies, including encyclicals, homilies, addresses, audiences, and related documents.
  - **Coverage profile:** Broader and deeper modern-document coverage than other sources.
  - **Website URL:** https://www.vatican.va/content/vatican/en.html

2. **Papal Encyclicals Collection (longer historical span; sparser documentation)**
  - **Source:** Papal Encyclicals Online
  - **Description:** Historical papal documents and directory references spanning a longer period, with comparatively sparser metadata/documentation in parts of the archive.
  - **Coverage profile:** Extends the temporal range of the corpus to older documents.
  - **Website URL:** https://www.papalencyclicals.net/document-directory

### Data Compilation

- **Scraping Methods:** Python-based web scraping using BeautifulSoup and requests libraries
- **Temporal Range:** 1215 CE to 2026 CE
- **Total Source Files in Repository (`data/raw/*.txt`):** 17,466
- **Documents in processed `LIBRARY.csv`:** 17,352

## Location

### Source Files Archive

- **GitHub Repository (Primary and only submission location):** https://github.com/rclayharris/encyclicals
  - Raw source files in `data/raw/`
  - Processed data files (CSV)
  - All notebooks (01_scraping, 02_pipeline, 03_exploration)
  - Source code (scraper.py, pipeline.py, analysis scripts)
  - Final report (final_report\DS5001_RCHarris_finalreport.pdf)
  - This README and MANIFEST

## Description

### Corpus Overview

This is a specialized collection of liturgical and doctrinal texts spanning over 800 years of Catholic ecclesiastical history (1215-2026). The primary analytical focus is **modern Catholic discourse** (1966-2026), encompassing papal encyclicals, pastoral letters, homilies, and formal addresses, with additional long-range historical material included from Papal Encyclicals Online.

### Subject Matter

- **Theological Content:** Catholic doctrinal teachings, ecclesiology, pastoral theology
- **Social Teachings:** Church positions on justice, peace, family, human dignity, social ethics
- **Historical Ecclesiastical Policy:** Decrees and canons from major church councils
- **Linguistic Variation:** Formal liturgical language, official church communications, evolving papal style across decades

### Analytical Focus Areas

- Topic modeling of doctrinal and pastoral concerns
- Sentiment and emotion analysis of pastoral discourse
- Lexical evolution and semantic drift in church teaching
- Stylistic variation across papal regimes and document types (encyclicals vs. speeches)

## Format

### Source File Format

**Format Type:** Plain text (UTF-8 encoded)  
**File Extension:** `.txt`  
**Line Endings:** Unix-style (`\n`)  
**Total Source Files:** 17,466

### File Naming Convention

Files follow a hierarchical naming pattern:

```
[category]__[document_type]__[title]__[date].txt
```

Examples:
- `papal_encyclicals__encyclical__evangelium_vitae__1995_03_25.txt`
- `church_councils__the_first_general_council_of_nicaea__325.txt`
- `papal_speeches__general_audience__urbi_et_orbi__2015_12_25.txt`

### Internal Document Structure

Each plaintext file contains:

1. **Header Section (if available)**
   - Document title
   - Pope/Council name
   - Date of issuance
   - Formal declaration or preamble

2. **Body Content**
   - Continuous prose text with paragraph breaks
   - Section headers where applicable
   - Numbered points or articles in formal documents

3. **No Markup:** Documents are provided as clean plaintext with minimal formatting

### Processed Data Formats

The corpus has been converted to STADM (Standard Text Analytic Data Model) and enhanced formats:

- **LIBRARY.csv** - Document metadata indexed by doc_id
  - Fields: pope, year, year_completed, document_type, source, length (in tokens)
  
- **TOKEN.csv** - Individual word tokens with linguistic annotation
  - Fields: token_id, doc_id, token_position, lemma, pos_tag, stopword_flag
  
- **VOCAB.csv** - Vocabulary statistics across entire corpus
  - Fields: term_str, n (frequency), tfidf_mean, tfidf_std
  
- **DOC_TOPICS.csv** - LDA topic model outputs (10 topics)
  - Fields: doc_id, topic_0 through topic_9 (topic concentrations per document)
  
- **TOPIC_TERMS.csv** - Top terms per topic with weights
  - Fields: term_str, topic_0 through topic_9 (term weights per topic)
  
- **DOC_PCA.csv** - Principal component analysis scores
  - Fields: doc_id, PC0, PC1, PC2, ... PC9
  
- **EMBEDDINGS.csv** - Word2Vec embeddings (100-dimensional)
  - Fields: term (vocabulary items), w2v_0 through w2v_99 (embedding dimensions)
  
- **DOC_SENTIMENT.csv** - Sentiment analysis output
  - Fields: doc_id, vader_compound, vader_pos, vader_neg, vader_neu

### Data Specifications

- **Character Encoding:** UTF-8
- **CSV Delimiter:** Comma (`,`)
- **CSV Quote Character:** Double quote (`"`)
- **Missing Values:** Represented as empty fields or `NaN` in numeric fields
- **Index Columns:** All tables include a primary key index (doc_id for document tables, term_str for vocabulary tables)
- **Unique Identifiers:** OCHO indexing applied where appropriate for relational integrity

## Metadata

### Corpus Statistics

- **Total Source Files (`data/raw/*.txt`):** 17,466
- **Total Documents (`LIBRARY.csv`):** 17,352
- **Total Tokens (`TOKEN.csv.gz`):** 28,425,031
- **Vocabulary Entries (`VOCAB.csv`):** 219,533
- **Date Range (`LIBRARY.year`):** 1215 CE - 2026 CE
- **Modern Era Focus (1966+):** 16,284 documents
- **Document Types:** Encyclicals, speeches, homilies, conciliar decrees, letters
- **Languages:** Primarily English, some Latin phrases

### Analytical Outputs

- **Principal Components:** 10 (explaining 3.03% of variance in TF-IDF space)
- **LDA Topics:** 10 topics derived from TF-IDF document vectors
- **Word2Vec Dimensions:** 100-dimensional embeddings trained on corpus vocabulary
- **Sentiment Model:** VADER (Valence Aware Dictionary and sEntiment Reasoner)

## Usage and License

- **Availability:** Source documents are public domain or permitted for academic use
- **Restrictions:** Check original sources for specific licensing terms
- **Attribution:** Proper citation of source materials and popes/councils recommended
- **Ethics:** Documents represent official church positions; analytical interpretations are independent research

## Contact and Questions

For questions about this corpus, its provenance, or analytical methodology:
- **Creator:** Robert Clay Harris
- **Email:** jbm2rt@virginia.edu
- **Institution:** University of Virginia, School of Data Science
- **Course:** DS 5001 - Text as Data
