"""
Rebuild F5 models from the existing f4 cache (17,352-doc full corpus).
- PCA, LDA (gensim LdaMulticore), Word2Vec
- Saves DOC_PCA, LOADINGS, explained_variance, DOC_TOPICS, TOPIC_TERMS, EMBEDDINGS
- Does NOT overwrite TFIDF_DTM.csv (kept at 553-doc English-subset version for figures)
"""
import sys
import os
from pathlib import Path

# Ensure we're working from the project root regardless of cwd
PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

import pickle
import pandas as pd

CACHE_DIR = Path("data/processed/cache")
PROCESSED = Path("data/processed")

print("Loading f4 cache...")
LIBRARY, TOKEN, VOCAB, TFIDF_DTM = pickle.load(open(CACHE_DIR / "f4.pkl", "rb"))
print(f"Loaded f4: LIBRARY={len(LIBRARY):,}, TOKEN={len(TOKEN):,}, TFIDF_DTM={TFIDF_DTM.shape}")

if __name__ == "__main__":
    from src.pipeline import build_f5_models, save_cache

    f5_results = build_f5_models(
        LIBRARY, TOKEN, VOCAB, TFIDF_DTM,
        n_components=10,
        n_topics=10,
        w2v_dim=100,
    )
    save_cache("f5", f5_results)
    print("f5 cache saved")

    # Save F5 table outputs — DO NOT overwrite TFIDF_DTM.csv
    for name, obj in f5_results.items():
        if name == "w2v_model":
            continue
        out = PROCESSED / f"{name}.csv"
        if isinstance(obj, pd.DataFrame):
            obj.to_csv(out)
            print(f"  Saved {name}.csv  shape={obj.shape}")
        elif isinstance(obj, pd.Series):
            obj.to_csv(out)
            print(f"  Saved {name}.csv  len={len(obj)}")

    print("Done.")
