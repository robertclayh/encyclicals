"""
Papal Encyclicals — Text Analytics Pipeline
DS 5001 - Exploratory Text Analytics Final Project

Converts raw scraped text (F0) through the Standard Text Analytic Data Model:
  F0 -> F1 (Machine Learning Corpus Format)
  F1 -> F2 (STADM: LIBRARY, TOKEN, VOCAB tables)
  F2 -> F3 (NLP annotations: POS, lemma, stopwords, sentiment)
  F3 -> F4 (TFIDF vectorization)
  F4 -> F5 (PCA, LDA, word2vec)

Usage:
    python src/pipeline.py                    # Run full pipeline
    python src/pipeline.py --step f2          # Run up through F2
    python src/pipeline.py --step f3          # Run up through F3
    python src/pipeline.py --english-only     # Only process English docs
"""

import os
import re
import json
import logging
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
INDEX_FILE = DATA_DIR / "encyclicals_index.json"


# ===========================================================================
# F0 -> F1: Machine Learning Corpus Format
# ===========================================================================

def build_f1_corpus(english_only: bool = True) -> pd.DataFrame:
    """
    Build the F1 corpus: a table of text chunks indexed by document hierarchy.

    Each row = one paragraph (the minimum discursive unit), with columns:
      doc_id, pope, title, year, para_num, para_text
    """
    logger.info("Building F1 corpus from raw text files...")

    with open(INDEX_FILE, "r", encoding="utf-8") as f:
        index = json.load(f)

    # Build a lookup from doc_id
    meta = {d["doc_id"]: d for d in index}

    rows = []
    for raw_file in sorted(RAW_DIR.glob("*.txt")):
        doc_id = raw_file.stem
        if doc_id not in meta:
            continue

        doc_meta = meta[doc_id]

        # Filter to English if requested
        if english_only and doc_meta.get("language", "") != "en":
            continue

        text = raw_file.read_text(encoding="utf-8")
        if len(text.strip()) < 100:
            continue

        # Split into paragraphs
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

        for i, para in enumerate(paragraphs):
            rows.append({
                "doc_id": doc_id,
                "pope": doc_meta.get("pope", "Unknown"),
                "title": doc_meta.get("title", ""),
                "year": doc_meta.get("year", ""),
                "para_num": i,
                "para_text": para,
            })

    corpus = pd.DataFrame(rows)
    logger.info(f"F1 corpus: {len(corpus)} paragraphs from "
                f"{corpus['doc_id'].nunique()} documents")
    return corpus


# ===========================================================================
# F1 -> F2: Standard Text Analytic Data Model (STADM)
# ===========================================================================

def build_f2_tables(corpus: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Tokenize the F1 corpus and produce STADM tables:
      - LIBRARY: document-level metadata
      - TOKEN: every token with position info (doc_id, para_num, sent_num, token_num)
      - VOCAB: unique terms with document frequency
    """
    logger.info("Building F2 STADM tables (LIBRARY, TOKEN, VOCAB)...")

    import nltk
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab", quiet=True)

    from nltk.tokenize import sent_tokenize, word_tokenize

    # --- LIBRARY table ---
    library_rows = []
    for doc_id, group in corpus.groupby("doc_id"):
        first = group.iloc[0]
        full_text = " ".join(group["para_text"].values)
        library_rows.append({
            "doc_id": doc_id,
            "pope": first["pope"],
            "title": first["title"],
            "year": first["year"],
            "n_paragraphs": len(group),
            "n_chars": len(full_text),
        })
    LIBRARY = pd.DataFrame(library_rows).set_index("doc_id")

    # --- TOKEN table ---
    token_rows = []
    token_idx = 0

    for _, row in tqdm(corpus.iterrows(), total=len(corpus), desc="Tokenizing"):
        sentences = sent_tokenize(row["para_text"])
        for sent_num, sentence in enumerate(sentences):
            words = word_tokenize(sentence)
            for pos_in_sent, token_str in enumerate(words):
                token_rows.append({
                    "token_id": token_idx,
                    "doc_id": row["doc_id"],
                    "para_num": row["para_num"],
                    "sent_num": sent_num,
                    "token_num": pos_in_sent,
                    "token_str": token_str,
                    "term_str": token_str.lower(),
                })
                token_idx += 1

    TOKEN = pd.DataFrame(token_rows).set_index("token_id")

    # --- VOCAB table ---
    # term_str, n (total count), df (document frequency)
    term_counts = TOKEN["term_str"].value_counts().rename("n")
    term_df = TOKEN.groupby("term_str")["doc_id"].nunique().rename("df")
    VOCAB = pd.DataFrame({"n": term_counts, "df": term_df})
    VOCAB.index.name = "term_str"
    VOCAB = VOCAB.sort_values("n", ascending=False)

    # Add n_docs for IDF calc
    n_docs = LIBRARY.shape[0]
    VOCAB["idf"] = np.log(n_docs / VOCAB["df"].replace(0, 1))

    # Update LIBRARY with token counts
    doc_token_counts = TOKEN.groupby("doc_id").size().rename("n_tokens")
    LIBRARY = LIBRARY.join(doc_token_counts)

    logger.info(f"LIBRARY: {len(LIBRARY)} documents")
    logger.info(f"TOKEN:   {len(TOKEN)} tokens")
    logger.info(f"VOCAB:   {len(VOCAB)} unique terms")

    return LIBRARY, TOKEN, VOCAB


# ===========================================================================
# F2 -> F3: NLP Annotations
# ===========================================================================

def build_f3_annotations(TOKEN: pd.DataFrame, VOCAB: pd.DataFrame,
                         LIBRARY: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Add NLP annotations to TOKEN and VOCAB:
      - POS tags
      - Lemmas
      - Stopword flags
      - Named entity labels (on TOKEN)
      - Sentiment scores (on VOCAB)
    """
    logger.info("Building F3 NLP annotations...")

    import nltk
    for resource in ["averaged_perceptron_tagger_eng", "wordnet", "stopwords",
                     "vader_lexicon", "omw-1.4"]:
        try:
            nltk.data.find(f"corpora/{resource}" if resource != "averaged_perceptron_tagger_eng"
                          else f"taggers/{resource}")
        except LookupError:
            nltk.download(resource, quiet=True)

    from nltk import pos_tag
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    from nltk.sentiment.vader import SentimentIntensityAnalyzer

    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()
    sid = SentimentIntensityAnalyzer()

    # --- POS tagging (batch by sentence for accuracy) ---
    logger.info("  POS tagging...")

    # Tag in sentence batches for proper context
    pos_tags = []
    for (doc_id, para_num, sent_num), group in tqdm(
        TOKEN.groupby(["doc_id", "para_num", "sent_num"]),
        desc="POS tagging"
    ):
        tokens = group["token_str"].tolist()
        tagged = pos_tag(tokens)
        pos_tags.extend([t[1] for t in tagged])

    TOKEN = TOKEN.copy()
    TOKEN["pos"] = pos_tags

    # --- Lemmatization ---
    logger.info("  Lemmatizing...")

    def get_wordnet_pos(tag):
        """Map POS tag to WordNet POS."""
        if tag.startswith("J"):
            return "a"
        elif tag.startswith("V"):
            return "v"
        elif tag.startswith("R"):
            return "r"
        return "n"

    TOKEN["lemma"] = [
        lemmatizer.lemmatize(row["term_str"], get_wordnet_pos(row["pos"]))
        for _, row in TOKEN.iterrows()
    ]

    # --- Stopword flag ---
    TOKEN["is_stop"] = TOKEN["term_str"].isin(stop_words)

    # --- Is alphabetic ---
    TOKEN["is_alpha"] = TOKEN["token_str"].str.isalpha()

    # --- Update VOCAB with POS and lemma info ---
    logger.info("  Updating VOCAB...")

    # Most common POS for each term
    vocab_pos = TOKEN.groupby("term_str")["pos"].agg(lambda x: x.mode().iloc[0] if len(x) > 0 else "")
    VOCAB = VOCAB.copy()
    VOCAB["pos"] = vocab_pos

    # Most common lemma for each term
    vocab_lemma = TOKEN.groupby("term_str")["lemma"].agg(lambda x: x.mode().iloc[0] if len(x) > 0 else "")
    VOCAB["lemma"] = vocab_lemma

    # Stopword flag
    VOCAB["is_stop"] = VOCAB.index.isin(stop_words)

    # --- Sentiment (VADER on terms) ---
    logger.info("  Computing VADER sentiment for vocab...")
    sentiments = []
    for term in VOCAB.index:
        scores = sid.polarity_scores(term)
        sentiments.append(scores)
    sent_df = pd.DataFrame(sentiments, index=VOCAB.index)
    VOCAB["vader_neg"] = sent_df["neg"]
    VOCAB["vader_neu"] = sent_df["neu"]
    VOCAB["vader_pos"] = sent_df["pos"]
    VOCAB["vader_compound"] = sent_df["compound"]

    # --- Document-level sentiment ---
    logger.info("  Computing document-level sentiment...")
    doc_sentiments = []
    for doc_id in LIBRARY.index:
        doc_tokens = TOKEN[TOKEN["doc_id"] == doc_id]
        doc_text = " ".join(doc_tokens["token_str"].values[:5000])  # limit for VADER
        scores = sid.polarity_scores(doc_text)
        doc_sentiments.append(scores)
    sent_doc_df = pd.DataFrame(doc_sentiments, index=LIBRARY.index)
    LIBRARY = LIBRARY.copy()
    LIBRARY["sentiment_neg"] = sent_doc_df["neg"]
    LIBRARY["sentiment_neu"] = sent_doc_df["neu"]
    LIBRARY["sentiment_pos"] = sent_doc_df["pos"]
    LIBRARY["sentiment_compound"] = sent_doc_df["compound"]

    logger.info("  F3 annotations complete")
    return LIBRARY, TOKEN, VOCAB


# ===========================================================================
# F3 -> F4: TFIDF Vectorization
# ===========================================================================

def build_f4_tfidf(TOKEN: pd.DataFrame, VOCAB: pd.DataFrame,
                   LIBRARY: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Compute TF and TFIDF values:
      - Add TF and TFIDF to TOKEN table
      - Add max_tfidf to VOCAB table
      - Create a document-term matrix (TFIDF_DTM)
    """
    logger.info("Building F4 TFIDF features...")

    n_docs = len(LIBRARY)

    # Term frequency per document
    tf = TOKEN.groupby(["doc_id", "term_str"]).size().rename("tf")
    tf = tf.reset_index()

    # Document lengths (for normalization)
    doc_lengths = TOKEN.groupby("doc_id").size().rename("doc_length")

    # Merge doc length and compute normalized TF
    tf = tf.merge(doc_lengths.reset_index(), on="doc_id")
    tf["tf_norm"] = tf["tf"] / tf["doc_length"]

    # Get IDF from VOCAB
    idf = VOCAB["idf"].to_dict()
    tf["idf"] = tf["term_str"].map(idf)
    tf["tfidf"] = tf["tf_norm"] * tf["idf"]

    # Create document-term matrix (pivot)
    logger.info("  Creating document-term matrix...")
    # Filter to meaningful terms for the DTM (alphabetic, not stopwords, appear in >1 doc)
    meaningful = VOCAB[(~VOCAB["is_stop"]) & (VOCAB.index.str.isalpha()) & (VOCAB["df"] > 1)]
    meaningful_terms = set(meaningful.index)

    tf_meaningful = tf[tf["term_str"].isin(meaningful_terms)]
    TFIDF_DTM = tf_meaningful.pivot_table(
        index="doc_id", columns="term_str", values="tfidf", fill_value=0
    )

    # Add max TFIDF per term to VOCAB
    max_tfidf = tf.groupby("term_str")["tfidf"].max()
    VOCAB = VOCAB.copy()
    VOCAB["max_tfidf"] = max_tfidf

    # Add TFIDF to TOKEN via merge
    TOKEN = TOKEN.copy()
    tf_lookup = tf.set_index(["doc_id", "term_str"])["tfidf"].to_dict()
    TOKEN["tfidf"] = [
        tf_lookup.get((row["doc_id"], row["term_str"]), 0.0)
        for _, row in TOKEN.iterrows()
    ]

    logger.info(f"  TFIDF DTM shape: {TFIDF_DTM.shape}")
    return LIBRARY, TOKEN, VOCAB, TFIDF_DTM


# ===========================================================================
# F4 -> F5: Unsupervised Models (PCA, LDA, word2vec)
# ===========================================================================

def build_f5_models(LIBRARY: pd.DataFrame, TOKEN: pd.DataFrame,
                    VOCAB: pd.DataFrame, TFIDF_DTM: pd.DataFrame,
                    n_components: int = 10, n_topics: int = 10,
                    w2v_dim: int = 100) -> dict:
    """
    Fit unsupervised models and add results to tables:
      - PCA: components table + loadings
      - LDA: topic distributions + topic-term weights
      - word2vec: term embeddings
    """
    logger.info("Building F5 unsupervised models...")

    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import LatentDirichletAllocation
    from sklearn.feature_extraction.text import CountVectorizer
    from gensim.models import Word2Vec

    results = {}

    # -----------------------------------------------------------------------
    # PCA on TFIDF matrix
    # -----------------------------------------------------------------------
    logger.info(f"  PCA with {n_components} components...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(TFIDF_DTM.values)

    n_comp = min(n_components, min(TFIDF_DTM.shape) - 1)
    pca = PCA(n_components=n_comp)
    doc_components = pca.fit_transform(X_scaled)

    # Document-component table
    pc_cols = [f"PC{i}" for i in range(n_comp)]
    DOC_PCA = pd.DataFrame(doc_components, index=TFIDF_DTM.index, columns=pc_cols)
    DOC_PCA.index.name = "doc_id"

    # Component loadings (terms x components)
    LOADINGS = pd.DataFrame(
        pca.components_.T,
        index=TFIDF_DTM.columns,
        columns=pc_cols,
    )
    LOADINGS.index.name = "term_str"

    # Explained variance
    explained_var = pd.Series(pca.explained_variance_ratio_, index=pc_cols, name="explained_variance")

    results["DOC_PCA"] = DOC_PCA
    results["LOADINGS"] = LOADINGS
    results["explained_variance"] = explained_var

    logger.info(f"    Explained variance: {explained_var.sum():.2%}")

    # -----------------------------------------------------------------------
    # LDA Topic Model
    # -----------------------------------------------------------------------
    logger.info(f"  LDA with {n_topics} topics...")

    # Build count matrix from raw text for LDA
    doc_texts = {}
    for doc_id in LIBRARY.index:
        doc_tokens = TOKEN[(TOKEN["doc_id"] == doc_id) &
                          (TOKEN["is_alpha"]) &
                          (~TOKEN["is_stop"])]
        doc_texts[doc_id] = " ".join(doc_tokens["lemma"].values)

    doc_ids = list(doc_texts.keys())
    texts = [doc_texts[d] for d in doc_ids]

    vectorizer = CountVectorizer(max_features=5000, min_df=2)
    count_matrix = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()

    n_top = min(n_topics, len(doc_ids) - 1)
    lda = LatentDirichletAllocation(
        n_components=n_top, random_state=42, max_iter=20
    )
    doc_topics = lda.fit_transform(count_matrix)

    # Document-topic table
    topic_cols = [f"topic_{i}" for i in range(n_top)]
    DOC_TOPICS = pd.DataFrame(doc_topics, index=doc_ids, columns=topic_cols)
    DOC_TOPICS.index.name = "doc_id"

    # Topic-term table
    TOPIC_TERMS = pd.DataFrame(
        lda.components_,
        index=topic_cols,
        columns=feature_names,
    ).T
    TOPIC_TERMS.index.name = "term_str"

    results["DOC_TOPICS"] = DOC_TOPICS
    results["TOPIC_TERMS"] = TOPIC_TERMS

    # Log top words per topic
    for i in range(n_top):
        top_words = TOPIC_TERMS[f"topic_{i}"].nlargest(10).index.tolist()
        logger.info(f"    Topic {i}: {', '.join(top_words)}")

    # -----------------------------------------------------------------------
    # Word2Vec embeddings
    # -----------------------------------------------------------------------
    logger.info(f"  word2vec with {w2v_dim} dimensions...")

    # Build sentences from TOKEN table
    sentences = []
    for (doc_id, para_num, sent_num), group in TOKEN.groupby(
        ["doc_id", "para_num", "sent_num"]
    ):
        tokens = group[group["is_alpha"]]["lemma"].tolist()
        if tokens:
            sentences.append(tokens)

    w2v_model = Word2Vec(
        sentences=sentences,
        vector_size=w2v_dim,
        window=5,
        min_count=5,
        workers=4,
        seed=42,
    )

    # Create embeddings table
    w2v_terms = list(w2v_model.wv.key_to_index.keys())
    embeddings = np.array([w2v_model.wv[t] for t in w2v_terms])
    emb_cols = [f"w2v_{i}" for i in range(w2v_dim)]
    EMBEDDINGS = pd.DataFrame(embeddings, index=w2v_terms, columns=emb_cols)
    EMBEDDINGS.index.name = "term_str"

    results["EMBEDDINGS"] = EMBEDDINGS
    results["w2v_model"] = w2v_model

    logger.info(f"    word2vec vocabulary: {len(w2v_terms)} terms")

    return results


# ===========================================================================
# Save / Load helpers
# ===========================================================================

def save_tables(LIBRARY, TOKEN, VOCAB, TFIDF_DTM=None, f5_results=None):
    """Save all tables to CSV."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Saving tables to CSV...")
    LIBRARY.to_csv(PROCESSED_DIR / "LIBRARY.csv")
    TOKEN.to_csv(PROCESSED_DIR / "TOKEN.csv")
    VOCAB.to_csv(PROCESSED_DIR / "VOCAB.csv")

    if TFIDF_DTM is not None:
        TFIDF_DTM.to_csv(PROCESSED_DIR / "TFIDF_DTM.csv")

    if f5_results:
        for name, df in f5_results.items():
            if isinstance(df, pd.DataFrame):
                df.to_csv(PROCESSED_DIR / f"{name}.csv")
            elif isinstance(df, pd.Series):
                df.to_csv(PROCESSED_DIR / f"{name}.csv")

    logger.info(f"All tables saved to {PROCESSED_DIR}/")


# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description="Papal Encyclicals Text Analytics Pipeline")
    parser.add_argument("--step", choices=["f1", "f2", "f3", "f4", "f5"],
                        default="f5", help="Run pipeline up through this step")
    parser.add_argument("--english-only", action="store_true", default=True,
                        help="Only process English-language documents")
    parser.add_argument("--n-components", type=int, default=10,
                        help="Number of PCA components")
    parser.add_argument("--n-topics", type=int, default=10,
                        help="Number of LDA topics")
    parser.add_argument("--w2v-dim", type=int, default=100,
                        help="word2vec embedding dimensions")
    args = parser.parse_args()

    steps = ["f1", "f2", "f3", "f4", "f5"]
    max_step_idx = steps.index(args.step)

    # F1
    corpus = build_f1_corpus(english_only=args.english_only)
    if max_step_idx < 1:
        corpus.to_csv(PROCESSED_DIR / "F1_CORPUS.csv", index=False)
        return

    # F2
    LIBRARY, TOKEN, VOCAB = build_f2_tables(corpus)
    if max_step_idx < 2:
        save_tables(LIBRARY, TOKEN, VOCAB)
        return

    # F3
    LIBRARY, TOKEN, VOCAB = build_f3_annotations(TOKEN, VOCAB, LIBRARY)
    if max_step_idx < 3:
        save_tables(LIBRARY, TOKEN, VOCAB)
        return

    # F4
    LIBRARY, TOKEN, VOCAB, TFIDF_DTM = build_f4_tfidf(TOKEN, VOCAB, LIBRARY)
    if max_step_idx < 4:
        save_tables(LIBRARY, TOKEN, VOCAB, TFIDF_DTM)
        return

    # F5
    f5_results = build_f5_models(
        LIBRARY, TOKEN, VOCAB, TFIDF_DTM,
        n_components=args.n_components,
        n_topics=args.n_topics,
        w2v_dim=args.w2v_dim,
    )
    save_tables(LIBRARY, TOKEN, VOCAB, TFIDF_DTM, f5_results)

    logger.info("Pipeline complete!")


if __name__ == "__main__":
    main()
