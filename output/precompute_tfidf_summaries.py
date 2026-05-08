"""
Precompute TFIDF-derived summaries needed by generate_figures.py.

Loads the f4 cache (which holds the full 17,352-doc TFIDF matrix in memory)
and saves two small summary files instead of the full matrix CSV:

  data/processed/TFIDF_FIG11.csv   — per-term mean TF-IDF for pre/post Vatican II
  data/processed/TFIDF_FIG12.csv   — for each social term: doc_ids containing it

Run this once after the f4 cache is available (or after any f4 rebuild).
"""
import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

import pickle
import pandas as pd

CACHE_DIR = Path("data/processed/cache")
PROCESSED = Path("data/processed")

print("Loading f4 cache (TFIDF_DTM in memory)...")
LIBRARY, TOKEN, VOCAB, TFIDF_DTM = pickle.load(open(CACHE_DIR / "f4.pkl", "rb"))
print(f"  LIBRARY={len(LIBRARY):,}  TFIDF_DTM={TFIDF_DTM.shape}")

# ── Year metadata ─────────────────────────────────────────────────────────────
LIB = pd.read_csv(PROCESSED / "LIBRARY.csv", index_col="doc_id")
LIB["year_num"] = pd.to_numeric(LIB["year"], errors="coerce")

# Corpus is English-only by construction (ENGLISH_ONLY=True in pipeline),
# so all documents are considered English-dominant.
lang_ok = pd.Series(True, index=LIB.index)

pre_vii  = LIB[(LIB["year_num"] <= 1961) & lang_ok].index
post_vii = LIB[(LIB["year_num"] >  1965) & lang_ok].index

print(f"  Pre-Vatican II docs: {len(pre_vii):,}")
print(f"  Post-Vatican II docs: {len(post_vii):,}")

# ── Figure 11: mean TF-IDF per term pre/post Vatican II ──────────────────────
print("Computing Fig 11 term means...")

# Filter to meaningful alpha terms (df >= 20, alpha, len > 2)
alpha_terms = [
    c for c in TFIDF_DTM.columns
    if c.isalpha() and len(c) > 2
    and c in VOCAB.index and VOCAB.loc[c, "df"] >= 20
]
print(f"  Eligible alpha terms (df>=20, len>2): {len(alpha_terms):,}")

eng_pre  = [d for d in pre_vii  if d in TFIDF_DTM.index]
eng_post = [d for d in post_vii if d in TFIDF_DTM.index]

pre_mean  = TFIDF_DTM.loc[eng_pre,  alpha_terms].mean()
post_mean = TFIDF_DTM.loc[eng_post, alpha_terms].mean()

fig11 = pd.DataFrame({"pre_mean": pre_mean, "post_mean": post_mean})
fig11.index.name = "term_str"
fig11.to_csv(PROCESSED / "TFIDF_FIG11.csv")
print(f"  Saved TFIDF_FIG11.csv  shape={fig11.shape}")
print(f"  Pre top 15:  {list(pre_mean.nlargest(15).index)}")
print(f"  Post top 15: {list(post_mean.nlargest(15).index)}")

# ── Figure 12: social term presence per document ─────────────────────────────
print("Computing Fig 12 term presence...")
social_terms = ["labor", "justice", "poor", "dignity", "social", "capital",
                "wage", "worker", "poverty", "ecology", "environment",
                "rights", "solidarity", "freedom"]

rows = []
for term in social_terms:
    if term in TFIDF_DTM.columns:
        docs_with = TFIDF_DTM.index[TFIDF_DTM[term] > 0].tolist()
        for doc_id in docs_with:
            rows.append({"term_str": term, "doc_id": doc_id})
    else:
        print(f"  WARNING: '{term}' not in TFIDF_DTM columns")

fig12 = pd.DataFrame(rows)
fig12.to_csv(PROCESSED / "TFIDF_FIG12.csv", index=False)
print(f"  Saved TFIDF_FIG12.csv  rows={len(fig12):,}")

# Store counts for reference
n_pre  = len(eng_pre)
n_post = len(eng_post)
meta = pd.Series({"n_pre_vatican_ii": n_pre, "n_post_vatican_ii": n_post,
                  "n_total": len(LIBRARY)})
meta.to_csv(PROCESSED / "TFIDF_FIG11_META.csv", header=False)
print(f"  n_pre={n_pre}  n_post={n_post}")
print("Done.")
