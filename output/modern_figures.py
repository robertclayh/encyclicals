"""
Modern Papal Corpus — Final Report Figure Generation
Clay Harris (jbm2rt@virginia.edu) / Text as Data / 2026-05-08

Generates all figures for the final report, including RQ1-RQ3 analysis and evidence plots:
  - RQ1: Modern papal signatures (PCA, topics, sentiment)
  - RQ2: Document-type differentiation (encyclicals vs. speeches)
  - RQ3: Temporal topic drift (raw vs. controlled)
  - PCA evidence: Explained variance profile
  - Word2Vec evidence: Seed-term similarity heatmap
"""

import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid", font_scale=1.1)

REPO = Path(__file__).resolve().parent.parent
PROCESSED = REPO / "data" / "processed"
FIG_DIR = REPO / "output" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def load_csv(stem, index_col=None):
    for ext in (".csv", ".csv.gz", ".csv.zip"):
        path = PROCESSED / f"{stem}{ext}"
        if path.exists():
            return pd.read_csv(path, index_col=index_col)
    raise FileNotFoundError(stem)


LIBRARY = load_csv("LIBRARY", index_col="doc_id")
DOC_PCA = load_csv("DOC_PCA", index_col="doc_id")
DOC_TOPICS = load_csv("DOC_TOPICS", index_col="doc_id")
TOPIC_TERMS = load_csv("TOPIC_TERMS", index_col="term_str")
DOC_SENTIMENT = load_csv("DOC_SENTIMENT", index_col="doc_id")
VOCAB = load_csv("VOCAB", index_col="term_str")
TFIDF_FIG12 = load_csv("TFIDF_FIG12")
EXPLAINED_VAR = load_csv("explained_variance")
EMBEDDINGS_IDX = load_csv("EMBEDDINGS", index_col=0)
METADATA = pd.DataFrame(json.loads((REPO / "data" / "encyclicals_index.json").read_text(encoding="utf-8"))).set_index("doc_id")

LIBRARY = LIBRARY.join(METADATA[["source", "document_type", "year"]].rename(columns={"year": "year_meta"}), how="left")
LIBRARY = LIBRARY.drop(columns=[c for c in DOC_SENTIMENT.columns if c in LIBRARY.columns], errors="ignore")
LIBRARY = LIBRARY.join(DOC_SENTIMENT, how="left")
LIBRARY["year_num"] = pd.to_numeric(LIBRARY["year_meta"], errors="coerce")
LIBRARY["year_num"] = LIBRARY["year_num"].fillna(pd.to_numeric(LIBRARY["year"], errors="coerce"))


def infer_year_from_doc_id(doc_id):
    if not isinstance(doc_id, str):
        return np.nan
    for token in re.findall(r"(?<!\d)(\d{8})(?!\d)", doc_id):
        ymd_year = int(token[:4])
        ymd_month = int(token[4:6])
        ymd_day = int(token[6:8])
        dmy_day = int(token[:2])
        dmy_month = int(token[2:4])
        dmy_year = int(token[4:8])
        ymd_valid = (1300 <= ymd_year <= 2100) and (1 <= ymd_month <= 12) and (1 <= ymd_day <= 31)
        dmy_valid = (1300 <= dmy_year <= 2100) and (1 <= dmy_month <= 12) and (1 <= dmy_day <= 31)
        if ymd_valid and not dmy_valid:
            return ymd_year
        if dmy_valid and not ymd_valid:
            return dmy_year
        if ymd_valid and dmy_valid:
            return ymd_year if 1800 <= ymd_year <= 2100 else dmy_year
    return np.nan


mask = LIBRARY["year_num"].isna()
LIBRARY.loc[mask, "year_num"] = pd.Index(LIBRARY.index[mask]).map(infer_year_from_doc_id)
LIBRARY["document_type_clean"] = LIBRARY["document_type"].fillna("unknown").str.strip().str.lower()

MODERN = LIBRARY[LIBRARY["year_num"] >= 1966].copy()
MODERN_VATICAN = MODERN[MODERN["source"].eq("vatican.va")].copy()
MODERN_SCOPE = MODERN_VATICAN.copy() if len(MODERN_VATICAN) > 0 else MODERN.copy()
MODERN_POPE = MODERN_SCOPE[MODERN_SCOPE["pope"].notna()].copy()

pope_first = MODERN_POPE.groupby("pope")["year_num"].min().sort_values()
pope_counts = MODERN_POPE["pope"].value_counts()
TOP_POPES = [p for p in pope_first.index if p in set(pope_counts.head(6).index)]
TOPIC_COLS = [c for c in DOC_TOPICS.columns if c.lower().startswith("topic")]
SENT_COL = "vader_compound"


NOISE_TERMS = {
    "u", "us", "amp", "http", "https", "www", "com", "org",
    "may", "dear", "holy", "one", "two", "three", "also",
    "said", "say", "says", "would", "could", "shall", "must",
    "like", "much", "many", "among", "within", "without",
}


def normalize_term(term):
    t = re.sub(r"[^a-z]", "", str(term).lower())
    if t.endswith("ing") and len(t) > 5:
        t = t[:-3]
    elif t.endswith("ed") and len(t) > 4:
        t = t[:-2]
    elif t.endswith("es") and len(t) > 4:
        t = t[:-2]
    elif t.endswith("s") and len(t) > 3:
        t = t[:-1]
    return t


def topic_terms_clean(topic_col, n_terms=6):
    ranked = TOPIC_TERMS[topic_col].sort_values(ascending=False).index.tolist()
    keep = []
    seen = set()
    for raw in ranked:
        raw_s = str(raw).strip().lower()
        if len(raw_s) < 2:
            continue
        if raw_s in NOISE_TERMS:
            continue
        if not re.search(r"[a-z]", raw_s):
            continue
        norm = normalize_term(raw_s)
        if len(norm) < 2 or norm in NOISE_TERMS:
            continue
        if norm in seen:
            continue
        seen.add(norm)
        keep.append(raw_s)
        if len(keep) >= n_terms:
            break
    if not keep:
        keep = TOPIC_TERMS[topic_col].sort_values(ascending=False).index.astype(str).tolist()[:n_terms]
    return keep


def topic_label(topic_col, n_terms=6):
    words = topic_terms_clean(topic_col, n_terms=n_terms)
    return f"{topic_col}: {', '.join(words)}"


def distinct_topic_labels(topic_cols, n_terms=5, max_global_repeats=1):
    ranked_terms = {
        t: topic_terms_clean(t, n_terms=max(20, n_terms * 6))
        for t in topic_cols
    }
    labels = {}
    used_topics_by_term = {}
    global_repeat_count = 0

    for topic in topic_cols:
        keep = []
        seen_local = set()

        for raw in ranked_terms[topic]:
            norm = normalize_term(raw)
            if not norm or norm in seen_local:
                continue

            prior_topics = used_topics_by_term.get(norm, set())
            is_repeat = len(prior_topics) > 0 and topic not in prior_topics
            if is_repeat and global_repeat_count >= max_global_repeats:
                continue

            keep.append(raw)
            seen_local.add(norm)
            used_topics_by_term.setdefault(norm, set()).add(topic)
            if is_repeat:
                global_repeat_count += 1
            if len(keep) >= n_terms:
                break

        if len(keep) < n_terms:
            for raw in ranked_terms[topic]:
                norm = normalize_term(raw)
                if not norm or norm in seen_local:
                    continue
                keep.append(raw)
                seen_local.add(norm)
                if len(keep) >= n_terms:
                    break

        labels[topic] = f"{topic}: {', '.join(keep)}"

    return labels


def robust_axis_limits(series, low_q=0.01, high_q=0.99, pad_frac=0.1, min_span=1.0):
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) == 0:
        return (-1.0, 1.0)
    lo, hi = s.quantile([low_q, high_q]).tolist()
    lo = float(lo)
    hi = float(hi)
    span = max(hi - lo, min_span)
    pad = span * pad_frac
    center = float(s.median())
    half = span / 2.0 + pad
    return (center - half, center + half)


def sentiment_ylim(series, low_q=0.01, high_q=0.99, pad=0.02):
    if len(series) == 0:
        return (-1.0, 1.0)
    q_low, q_high = series.quantile([low_q, high_q]).tolist()
    y0 = max(-1.0, float(q_low) - pad)
    y1 = min(1.0, float(q_high) + pad)
    return (y0, y1)


# Figure A1: Zipf-like frequency structure
zipf_counts = pd.to_numeric(VOCAB["n"], errors="coerce").dropna()
zipf_counts = zipf_counts[zipf_counts > 0].sort_values(ascending=False).reset_index(drop=True)
zipf_ranks = np.arange(1, len(zipf_counts) + 1)
fig, ax = plt.subplots(figsize=(10, 6))
ax.loglog(zipf_ranks, zipf_counts, color="#1f77b4", linewidth=1.8)
ax.set_title("Exploratory: Zipf-Like Frequency Structure")
ax.set_xlabel("Term rank (log)")
ax.set_ylabel("Term frequency n (log)")
ax.grid(alpha=0.25)
plt.tight_layout()
fig.savefig(FIG_DIR / "modern_explore_zipf.png")
plt.close(fig)


# Figure A2: TF-IDF salience via frequent top-TFIDF terms per document
top_tfidf_terms = (
    TFIDF_FIG12["term_str"].astype(str).value_counts().head(15).sort_values(ascending=True)
)
fig, ax = plt.subplots(figsize=(10, 7))
ax.barh(top_tfidf_terms.index, top_tfidf_terms.values, color="#2a9d8f")
ax.set_title("Exploratory: Terms Most Often Top-TFIDF in Documents")
ax.set_xlabel("Count of documents where term is top-TFIDF")
ax.set_ylabel("Term")
ax.grid(axis="x", alpha=0.25)
plt.tight_layout()
fig.savefig(FIG_DIR / "modern_explore_tfidf_salience.png")
plt.close(fig)


# Figure A3: Sentiment distributions by major document type
doc_type_sent = MODERN_SCOPE.dropna(subset=[SENT_COL]).copy()
top_types = doc_type_sent["document_type_clean"].value_counts().head(5).index.tolist()
doc_type_sent = doc_type_sent[doc_type_sent["document_type_clean"].isin(top_types)]
fig, ax = plt.subplots(figsize=(12, 6))
sns.violinplot(
    data=doc_type_sent,
    x="document_type_clean",
    y=SENT_COL,
    order=top_types,
    color="#bde0fe",
    inner=None,
    cut=0,
    linewidth=0.8,
    ax=ax,
)
sns.boxplot(
    data=doc_type_sent,
    x="document_type_clean",
    y=SENT_COL,
    order=top_types,
    color="#219ebc",
    width=0.22,
    showfliers=False,
    boxprops={"alpha": 0.75},
    ax=ax,
)
ax.set_ylim(*sentiment_ylim(doc_type_sent[SENT_COL]))
ax.set_title("Exploratory: VADER Compound by Major Modern Document Type")
ax.set_xlabel("Document type")
ax.set_ylabel("VADER compound (zoomed high-end range)")
ax.tick_params(axis="x", rotation=25)
ax.grid(alpha=0.25)
ax.text(
    0.01,
    0.02,
    f"Full-range min/max: {doc_type_sent[SENT_COL].min():.3f} to {doc_type_sent[SENT_COL].max():.3f}",
    transform=ax.transAxes,
    fontsize=9,
    color="#3d405b",
)
plt.tight_layout()
fig.savefig(FIG_DIR / "modern_explore_sentiment_types.png")
plt.close(fig)

# Figure 1: modern pope PCA
pca_df = DOC_PCA.join(MODERN_POPE[["pope"]], how="inner")
pca_df = pca_df[pca_df["pope"].isin(TOP_POPES)]
xlim = (-0.5, 0.25)
ylim = (-0.5, 0.1)
inlier_mask = pca_df["PC0"].between(xlim[0], xlim[1]) & pca_df["PC1"].between(ylim[0], ylim[1])
pca_plot = pca_df.loc[inlier_mask].copy()
if len(pca_plot) < 250:
    pca_plot = pca_df.copy()
fig, ax = plt.subplots(figsize=(12, 8))
colors = sns.color_palette("tab10", n_colors=max(1, len(TOP_POPES)))
legend_handles = []
for pope, color in zip(TOP_POPES, colors):
    m = pca_plot["pope"] == pope
    subset = pca_plot.loc[m, ["PC0", "PC1"]].dropna()
    if len(subset) == 0:
        continue
    sample_n = min(650, len(subset))
    sample = subset.sample(n=sample_n, random_state=42)
    ax.scatter(sample["PC0"], sample["PC1"], s=14, alpha=0.14, color=color, edgecolors="none")
    if len(subset) >= 80 and subset["PC0"].std() > 1e-9 and subset["PC1"].std() > 1e-9:
        try:
            sns.kdeplot(
                data=subset,
                x="PC0",
                y="PC1",
                levels=4,
                linewidths=1.15,
                fill=False,
                color=color,
                alpha=0.9,
                thresh=0.08,
                ax=ax,
            )
        except ValueError:
            pass
    centroid = subset.mean()
    ax.scatter(
        centroid["PC0"],
        centroid["PC1"],
        marker="X",
        s=140,
        color=color,
        edgecolors="black",
        linewidths=0.6,
        zorder=4,
    )
    legend_handles.append(Line2D([0], [0], marker="X", color=color, markersize=8, linewidth=1.4, label=pope))
ax.set_xlim(*xlim)
ax.set_ylim(*ylim)
ax.set_title("Modern Papal Signatures in PCA Space")
ax.set_xlabel("PC0")
ax.set_ylabel("PC1")
ax.legend(handles=legend_handles, title="Centroid by pope", loc="best", fontsize=9)
ax.grid(alpha=0.3)
ax.text(
    0.01,
    0.02,
    "Axes use 1st-99th percentile inlier bounds of modern-pope PCA scores",
    transform=ax.transAxes,
    fontsize=9,
    color="#3d405b",
)
plt.tight_layout()
fig.savefig(FIG_DIR / "modern_rq1_pca.png")
plt.close(fig)

# Figure 2: modern pope topic heatmap
pt = DOC_TOPICS.join(MODERN_POPE[["pope"]], how="inner").groupby("pope")[TOPIC_COLS].mean().loc[TOP_POPES]
renamed = distinct_topic_labels(TOPIC_COLS, n_terms=4, max_global_repeats=1)
fig, ax = plt.subplots(figsize=(12, 6))
sns.heatmap(pt.rename(columns=renamed), cmap="YlOrRd", annot=True, fmt=".2f", linewidths=0.3, ax=ax)
ax.set_title("Modern Papal Topic Profiles")
ax.set_xlabel("Topic (noise-filtered top terms)")
ax.set_ylabel("Pope")
plt.tight_layout()
fig.savefig(FIG_DIR / "modern_rq1_topics.png")
plt.close(fig)

# Figure 3: document type sentiment
doc_type_sent = MODERN_SCOPE.dropna(subset=[SENT_COL]).copy()
top_types = doc_type_sent["document_type_clean"].value_counts().head(5).index.tolist()
doc_type_sent = doc_type_sent[doc_type_sent["document_type_clean"].isin(top_types)]
fig, ax = plt.subplots(figsize=(12, 6))
sns.violinplot(
    data=doc_type_sent,
    x="document_type_clean",
    y=SENT_COL,
    order=top_types,
    color="#ffd166",
    inner=None,
    cut=0,
    linewidth=0.8,
    ax=ax,
)
sns.boxplot(
    data=doc_type_sent,
    x="document_type_clean",
    y=SENT_COL,
    order=top_types,
    color="#fb8500",
    width=0.22,
    showfliers=False,
    boxprops={"alpha": 0.8},
    ax=ax,
)
ax.set_ylim(*sentiment_ylim(doc_type_sent[SENT_COL]))
ax.set_title("Modern Document-Type Sentiment Profiles")
ax.set_xlabel("Document type")
ax.set_ylabel("VADER compound (zoomed high-end range)")
ax.tick_params(axis="x", rotation=25)
ax.grid(alpha=0.25)
ax.text(
    0.01,
    0.02,
    f"Full-range min/max: {doc_type_sent[SENT_COL].min():.3f} to {doc_type_sent[SENT_COL].max():.3f}",
    transform=ax.transAxes,
    fontsize=9,
    color="#3d405b",
)
plt.tight_layout()
fig.savefig(FIG_DIR / "modern_rq2_doc_type_sentiment.png")
plt.close(fig)

# Figure 4: modern encyclical vs speech topic contrast
enc_speech = MODERN_SCOPE[MODERN_SCOPE["document_type_clean"].isin(["encyclical", "speech"])][
    ["document_type_clean", "pope"]
].copy()
enc_speech_topics = DOC_TOPICS.join(enc_speech, how="inner")
topic_means = enc_speech_topics.groupby("document_type_clean")[TOPIC_COLS].mean()
topic_delta = (topic_means.loc["encyclical"] - topic_means.loc["speech"]).sort_values(
    key=lambda s: s.abs(), ascending=False
)
top_delta = topic_delta.head(6).sort_values()
delta_labels = distinct_topic_labels(top_delta.index.tolist(), n_terms=4, max_global_repeats=1)
fig, ax = plt.subplots(figsize=(12, 6))
colors = ["#e76f51" if v > 0 else "#457b9d" for v in top_delta.values]
ax.barh([delta_labels[t] for t in top_delta.index], top_delta.values, color=colors)
ax.axvline(0.0, color="black", linewidth=1)
ax.set_title("RQ2: Topic Weight Difference (Encyclicals - Speeches)")
ax.set_xlabel("Mean topic weight difference (>0 encyclical-heavy, <0 speech-heavy)")
ax.set_ylabel("Topic")
ax.grid(axis="x", alpha=0.25)
legend_handles = [
    Patch(facecolor="#e76f51", edgecolor="none", label="Encyclical-heavy"),
    Patch(facecolor="#457b9d", edgecolor="none", label="Speech-heavy"),
]
ax.legend(handles=legend_handles, loc="lower right", frameon=True)
plt.tight_layout()
fig.savefig(FIG_DIR / "modern_rq2_encyclical_vs_speech_topics.png")
plt.close(fig)

# Figure 5: modern topic evolution
time_df = DOC_TOPICS.join(MODERN[["year_num"]], how="inner").dropna(subset=["year_num"]).copy()
time_df["decade"] = (time_df["year_num"] // 10 * 10).astype(int)
decade_topics = time_df.groupby("decade")[TOPIC_COLS].mean()
topic_spread = (decade_topics.max(axis=0) - decade_topics.min(axis=0)).sort_values(ascending=False)
plot_topics = topic_spread.head(6).index.tolist()
drift_labels = distinct_topic_labels(plot_topics, n_terms=4, max_global_repeats=1)
fig, ax = plt.subplots(figsize=(13, 9))
palette = sns.color_palette("tab10", n_colors=len(plot_topics))
for topic, color in zip(plot_topics, palette):
    label = drift_labels[topic]
    ax.plot(decade_topics.index, decade_topics[topic], marker="o", linewidth=2.2, label=label, color=color)
ax.set_title("RQ3: Modern Topic Drift by Decade (Most Dynamic Topics)")
ax.set_xlabel("Decade")
ax.set_ylabel("Mean topic weight")
ax.grid(alpha=0.3)
ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=True, title="Topic label")
plt.tight_layout()
fig.savefig(FIG_DIR / "modern_rq3_topic_drift.png")
plt.close(fig)

# Figure 6: RQ3 raw vs pope+genre-controlled decade-range comparison
rq3_df = DOC_TOPICS.join(
    MODERN_SCOPE[["year_num", "pope", "document_type_clean"]],
    how="inner",
).dropna(subset=["year_num"]).copy()
rq3_df = rq3_df[rq3_df["pope"].notna()]
rq3_df["decade"] = (rq3_df["year_num"] // 10 * 10).astype(int)

raw_means = rq3_df.groupby("decade")[TOPIC_COLS].mean()
raw_range = (raw_means.max(axis=0) - raw_means.min(axis=0)).sort_values(ascending=False)

for t in TOPIC_COLS:
    rq3_df[t + "_resid"] = rq3_df[t] - rq3_df.groupby(["pope", "document_type_clean"])[t].transform("mean")
resid_cols = [t + "_resid" for t in TOPIC_COLS]
resid_means = rq3_df.groupby("decade")[resid_cols].mean()
resid_range = resid_means.max(axis=0) - resid_means.min(axis=0)
resid_range.index = [c.replace("_resid", "") for c in resid_range.index]
resid_range = resid_range.sort_values(ascending=False)

selected_topics = []
for t in list(raw_range.head(6).index) + list(resid_range.head(6).index):
    if t not in selected_topics:
        selected_topics.append(t)
selected_topics = selected_topics[:8]

compare_df = pd.DataFrame(
    {
        "topic": selected_topics,
        "raw_range": [float(raw_range.get(t, np.nan)) for t in selected_topics],
        "controlled_range": [float(resid_range.get(t, np.nan)) for t in selected_topics],
    }
).sort_values("raw_range", ascending=True)

fig, ax = plt.subplots(figsize=(12, 7))
y = np.arange(len(compare_df))
h = 0.38
ax.barh(y + h / 2, compare_df["raw_range"], height=h, color="#e76f51", label="Raw decade range")
ax.barh(y - h / 2, compare_df["controlled_range"], height=h, color="#457b9d", label="Controlled decade range")
ax.set_yticks(y)
ax.set_yticklabels([topic_label(t, n_terms=3) for t in compare_df["topic"]])
ax.set_xlabel("Decade range in mean topic weight")
ax.set_title("RQ3: Raw vs Pope+Genre-Controlled Topic Drift")
ax.grid(axis="x", alpha=0.25)
ax.legend(loc="lower right", frameon=True)
plt.tight_layout()
fig.savefig(FIG_DIR / "modern_rq3_raw_vs_control.png")
plt.close(fig)

# Figure 8: PCA explained variance (scree + cumulative)
ev_col = "explained_variance"
if ev_col not in EXPLAINED_VAR.columns:
    numeric_cols = EXPLAINED_VAR.select_dtypes(include="number").columns.tolist()
    ev_col = numeric_cols[0]
ev = pd.to_numeric(EXPLAINED_VAR[ev_col], errors="coerce").dropna().reset_index(drop=True)
pc_ids = np.arange(1, len(ev) + 1)
ev10 = float(ev.iloc[:10].sum()) if len(ev) >= 10 else float(ev.sum())

fig, ax1 = plt.subplots(figsize=(11, 6))
ax1.bar(pc_ids, ev.values, color="#4c78a8", alpha=0.8, label="Per-component explained variance")
ax1.set_xlabel("Principal component")
ax1.set_ylabel("Explained variance ratio")
ax1.set_title("PCA Explained Variance Profile")
ax1.grid(axis="y", alpha=0.25)
ax1.set_ylim(0, ev.values.max() * 1.4)

ax2 = ax1.twinx()
cum = np.cumsum(ev.values)
ax2.plot(pc_ids, cum, color="#f28e2b", linewidth=2.3, marker="o", markersize=3, label="Cumulative explained variance")
ax2.set_ylabel("Cumulative explained variance ratio")
ax2.set_ylim(0, cum.max() * 1.4)

if len(pc_ids) >= 10:
    ax2.axvline(10, color="#6c757d", linestyle="--", linewidth=1)
    ax2.text(10.2, min(0.98, cum[9] + 0.03), f"Top 10 = {ev10*100:.2f}%", fontsize=9, color="#2f3e46")

handles1, labels1 = ax1.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper left", frameon=True)
plt.tight_layout()
fig.savefig(FIG_DIR / "modern_pca_explained_variance.png")
plt.close(fig)

# Figure 9: Word2Vec seed-term cosine similarity heatmap
# Categories: social teaching (labor, justice, social), church structure (church, bishop, magisterium), modern world (modern, culture, society)
seed_terms = ["labor", "justice", "social", "church", "bishop", "magisterium", "modern", "culture", "society"]
emb_num = EMBEDDINGS_IDX.select_dtypes(include="number").copy()
emb_num.index = emb_num.index.astype(str)
norms = np.linalg.norm(emb_num.values, axis=1)
valid = norms > 0
emb_num = emb_num.iloc[valid]
norms = norms[valid]

present = [t for t in seed_terms if t in set(emb_num.index)]
if len(present) >= 4:
    E = emb_num.loc[present].values
    E = E / np.linalg.norm(E, axis=1, keepdims=True)
    sim = E @ E.T
    sim_df = pd.DataFrame(sim, index=present, columns=present)

    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    sns.heatmap(sim_df, cmap="YlGnBu", annot=True, fmt=".2f", vmin=-0.2, vmax=1.0, linewidths=0.3, ax=ax)
    ax.set_title("Word2Vec Seed-Term Cosine Similarity")
    ax.set_xlabel("Seed term")
    ax.set_ylabel("Seed term")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "modern_w2v_seed_similarity.png")
    plt.close(fig)

# Figure 7: PCA component-topic correlation map (inlier-trimmed)
pca_topic_df = DOC_PCA.join(MODERN_SCOPE[["pope"]], how="inner").join(DOC_TOPICS, how="inner")
pca_topic_df = pca_topic_df[pca_topic_df["pope"].notna()].copy()
pcs = ["PC0", "PC1", "PC2"]
pca_topics_focus = ["topic_3", "topic_9", "topic_7", "topic_2", "topic_6", "topic_4", "topic_8"]

corr_map = pd.DataFrame(index=pcs, columns=pca_topics_focus, dtype=float)
for pc in pcs:
    d = pca_topic_df[[pc] + pca_topics_focus].dropna().copy()
    q0, q1 = d[pc].quantile([0.01, 0.99]).tolist()
    d = d[d[pc].between(q0, q1)]
    corr_map.loc[pc] = d[pca_topics_focus].corrwith(d[pc]).values

display_labels = distinct_topic_labels(pca_topics_focus, n_terms=3, max_global_repeats=1)
fig, ax = plt.subplots(figsize=(12, 4.8))
sns.heatmap(
    corr_map,
    cmap="RdBu_r",
    center=0.0,
    annot=True,
    fmt=".2f",
    linewidths=0.3,
    cbar_kws={"label": "Correlation with principal component"},
    ax=ax,
)
ax.set_title("PCA Interpretation: Component-Topic Correlations (1st-99th percentile trimmed)")
ax.set_xlabel("Topic")
ax.set_ylabel("Principal component")
ax.set_xticklabels([display_labels[t] for t in pca_topics_focus], rotation=25, ha="right")
plt.tight_layout()
fig.savefig(FIG_DIR / "modern_pca_topic_corr.png")
plt.close(fig)

print("Saved modern figures to", FIG_DIR)
