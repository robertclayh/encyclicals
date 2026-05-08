"""
Regenerate all 15 figures for the Final Project.

Changes vs previous version:
  - All pope-level figures use chronological ordering
  - Language-contaminated docs (Italian/Latin) flagged via topic_4+topic_5
  - Vocabulary shift figure (fig11) filtered to English-dominant docs only
  - Larger fonts, cleaner layouts

Run: python output/generate_figures.py
"""

import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch
from sklearn.manifold import TSNE

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO      = Path(__file__).resolve().parent.parent
PROCESSED = REPO / "data" / "processed"
FIGURES   = REPO / "output" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

# ── Style ──────────────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", font_scale=1.2)
plt.rcParams.update({
	"figure.dpi": 130,
	"savefig.bbox": "tight",
	"axes.titlesize": 13,
	"axes.labelsize": 12,
	"xtick.labelsize": 10,
	"ytick.labelsize": 10,
	"legend.fontsize": 10,
})

# ── Load all tables ────────────────────────────────────────────────────────────
print("Loading data...")
LIBRARY    = pd.read_csv(PROCESSED / "LIBRARY.csv",    index_col="doc_id")
VOCAB      = pd.read_csv(PROCESSED / "VOCAB.csv",      index_col="term_str")
TFIDF_DTM  = pd.read_csv(PROCESSED / "TFIDF_DTM.csv",  index_col="doc_id")
DOC_PCA    = pd.read_csv(PROCESSED / "DOC_PCA.csv",    index_col="doc_id")
DOC_TOPICS = pd.read_csv(PROCESSED / "DOC_TOPICS.csv", index_col="doc_id")
LOADINGS   = pd.read_csv(PROCESSED / "LOADINGS.csv",   index_col="term_str")
TOPIC_TERMS = pd.read_csv(PROCESSED / "TOPIC_TERMS.csv", index_col="term_str")
EMBEDDINGS = pd.read_csv(PROCESSED / "EMBEDDINGS.csv", index_col="term_str")
EV_RAW     = pd.read_csv(PROCESSED / "explained_variance.csv", header=None)
ev_vals    = EV_RAW.iloc[1:, 1].astype(float).values
explained_var = pd.Series(ev_vals, index=[f"PC{i}" for i in range(len(ev_vals))])

LIBRARY["year_num"] = pd.to_numeric(LIBRARY["year"], errors="coerce")

# ── Language contamination flag ────────────────────────────────────────────────
# topic_4 top terms: et, de, la, le, ad, est (Latin/French)
# topic_5 top terms: di, che, la, il, per, non (Italian)
# Docs where topic_4 + topic_5 > 0.5 are primarily non-English originals
lang_contaminated = (DOC_TOPICS["topic_4"] + DOC_TOPICS["topic_5"]) > 0.5
LIBRARY["lang_ok"] = ~lang_contaminated.reindex(LIBRARY.index, fill_value=False)
print(f"Language-contaminated docs: {lang_contaminated.sum()} of {len(LIBRARY)}")

pope_lang = LIBRARY.groupby("pope")["lang_ok"].mean()
contaminated_popes = set(pope_lang[pope_lang < 0.5].index)
print("Popes predominantly non-English:", contaminated_popes)

# ── Chronological pope order ──────────────────────────────────────────────────
pope_first_year = LIBRARY.groupby("pope")["year_num"].min().sort_values()
pope_counts     = LIBRARY["pope"].value_counts()

# All popes with >= 2 docs, ordered chronologically
sig_popes_chron = [p for p in pope_first_year.index if pope_counts.get(p, 0) >= 2]

# Top 9 by doc count, in chronological order
top9_by_count = set(pope_counts.head(9).index)
top9_chron    = [p for p in sig_popes_chron if p in top9_by_count]

modern_popes_chron = [
	"Pope St. John XXIII",
	"Pope Paul VI",
	"Pope St. John Paul II",
	"Pope Benedict XVI",
	"Pope Francis",
]


def short_name(p):
	return p.replace("Pope ", "").replace("St. ", "").replace("Bl. ", "")


def period_of(y):
	if pd.isna(y):   return "Date unknown"
	if y < 1891:     return "Pre-Modern (<=1890)"
	if y < 1962:     return "Social Teaching (1891-1961)"
	if y <= 1965:    return "Vatican II (1962-65)"
	if y <= 2000:    return "Post-Vatican II (1966-2000)"
	return "Contemporary (2001+)"


LIBRARY["era"] = LIBRARY["year_num"].apply(period_of)
era_order  = ["Pre-Modern (<=1890)", "Social Teaching (1891-1961)",
			  "Vatican II (1962-65)", "Post-Vatican II (1966-2000)",
			  "Contemporary (2001+)", "Date unknown"]
era_colors = ["#795548", "#f57c00", "#fbc02d", "#388e3c", "#1565c0", "#9e9e9e"]

pre_vii  = LIBRARY[LIBRARY["year_num"] < 1962].index
post_vii = LIBRARY[LIBRARY["year_num"] > 1965].index
lib_dated = LIBRARY.dropna(subset=["year_num"])

topic_short = {col: ", ".join(TOPIC_TERMS[col].nlargest(3).index.tolist())
			   for col in TOPIC_TERMS.columns}
artifact_topics = ["topic_3", "topic_4", "topic_5"]


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1: Documents per pope (horizontal bar, chronological)
# ─────────────────────────────────────────────────────────────────────────────
print("Fig 1: docs per pope...")
popes_for_bar = [p for p in sig_popes_chron if pope_counts.get(p, 0) >= 2]
bar_vals      = pd.Series({p: pope_counts[p] for p in popes_for_bar})
bar_labels    = [short_name(p) for p in popes_for_bar]
bar_colors    = ["#c62828" if p in modern_popes_chron else "steelblue"
				 for p in popes_for_bar]

fig, ax = plt.subplots(figsize=(12, 10))
bars = ax.barh(range(len(popes_for_bar)), bar_vals.values,
			   color=bar_colors, alpha=0.82)
for b, p in zip(bars, popes_for_bar):
	if p in contaminated_popes:
		b.set_hatch("//")
		b.set_edgecolor("gray")
ax.set_yticks(range(len(popes_for_bar)))
ax.set_yticklabels(bar_labels, fontsize=9)
ax.set_xlabel("Number of Documents")
ax.set_title("Documents per Pope (chronological)\n"
			 "Hatched = predominantly non-English documents", fontweight="bold")
ax.invert_yaxis()
ax.grid(axis="x", alpha=0.4)
ax.legend(handles=[
	Patch(facecolor="steelblue", label="Pre-Vatican II pope"),
	Patch(facecolor="#c62828",   label="Post-1958 pope"),
	Patch(facecolor="steelblue", hatch="//", edgecolor="gray",
		  label="Primarily non-English docs"),
], fontsize=9, loc="lower right")
plt.tight_layout()
fig.savefig(FIGURES / "fig01_docs_per_pope.png")
plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2: Corpus temporal distribution
# ─────────────────────────────────────────────────────────────────────────────
print("Fig 2: corpus over time...")
decade_counts = lib_dated.groupby(
	(lib_dated["year_num"] // 10 * 10).astype(int)).size()

fig, axes = plt.subplots(1, 2, figsize=(16, 5))
decade_counts.plot(kind="bar", ax=axes[0], color="steelblue", alpha=0.8)
axes[0].set_xlabel("Decade")
axes[0].set_ylabel("Documents")
axes[0].set_title("Documents per Decade")
axes[0].tick_params(axis="x", rotation=45)
axes[0].grid(axis="y", alpha=0.4)

axes[1].scatter(lib_dated["year_num"], lib_dated["n_tokens"],
				alpha=0.5, s=30, color="steelblue", edgecolors="none")
axes[1].set_xlabel("Year")
axes[1].set_ylabel("Token Count")
axes[1].set_title("Document Length Over Time")
axes[1].axvline(1891, color="orange", linestyle="--", linewidth=1.2,
				alpha=0.8, label="Rerum Novarum (1891)")
axes[1].axvline(1965, color="darkred", linestyle="--", linewidth=1.2,
				alpha=0.8, label="Vatican II ends (1965)")
axes[1].legend(fontsize=9)
axes[1].grid(alpha=0.3)

plt.suptitle("Corpus Temporal Distribution", fontweight="bold", fontsize=14)
plt.tight_layout()
fig.savefig(FIGURES / "fig02_corpus_over_time.png")
plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3: Zipf's Law
# ─────────────────────────────────────────────────────────────────────────────
print("Fig 3: Zipf...")
vocab_sorted = VOCAB["n"].sort_values(ascending=False).reset_index(drop=True)
ranks = np.arange(1, len(vocab_sorted) + 1)

fig, ax = plt.subplots(figsize=(10, 6))
ax.loglog(ranks[:8000], vocab_sorted.values[:8000],
		  color="steelblue", linewidth=1.5)
ax.set_xlabel("Rank (log)")
ax.set_ylabel("Frequency (log)")
ax.set_title("Term Frequency Distribution (Zipf's Law)")
ax.grid(True, alpha=0.3, which="both")

term_list = VOCAB["n"].sort_values(ascending=False).index.tolist()
for term in ["church", "god", "christ", "life", "faith", "holy",
			 "human", "love", "people", "world", "man"]:
	if term in term_list:
		r = term_list.index(term) + 1
		f = VOCAB.loc[term, "n"]
		ax.annotate(term, xy=(r, f), fontsize=8.5, color="darkred",
					xytext=(r * 1.6, f * 0.7),
					arrowprops=dict(arrowstyle="-", color="gray", lw=0.7))
plt.tight_layout()
fig.savefig(FIGURES / "fig03_zipf.png")
plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4: Sentiment by era and over time
# ─────────────────────────────────────────────────────────────────────────────
print("Fig 4: sentiment...")
sc = "sentiment_compound"
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

era_data = [LIBRARY[LIBRARY["era"] == e][sc].dropna()
			for e in era_order if e != "Date unknown"]
era_lbls = [e for e in era_order if e != "Date unknown"]
bp = axes[0].boxplot(era_data, patch_artist=True,
					 labels=[e.split(" (")[0] for e in era_lbls],
					 medianprops=dict(color="black", linewidth=2))
for patch, col in zip(bp["boxes"], era_colors[:5]):
	patch.set_facecolor(col)
	patch.set_alpha(0.75)
axes[0].axhline(0, color="gray", linestyle="--", alpha=0.5)
axes[0].set_ylabel("VADER Compound Score")
axes[0].set_title("Sentiment Distribution by Era")
axes[0].tick_params(axis="x", rotation=25)
axes[0].grid(axis="y", alpha=0.3)

lib_sent = lib_dated[[sc, "year_num"]].dropna()
sc2 = axes[1].scatter(lib_sent["year_num"], lib_sent[sc],
					  c=lib_sent["year_num"], cmap="RdYlGn",
					  alpha=0.45, s=30, edgecolors="none")
plt.colorbar(sc2, ax=axes[1], label="Year")
roll = (lib_sent.sort_values("year_num")
		.set_index("year_num")[sc]
		.rolling(20, center=True).mean())
axes[1].plot(roll.index, roll.values, color="black", linewidth=2.5,
			 label="20-doc rolling mean")
axes[1].axvline(1965, color="darkred", linestyle="--", linewidth=1.5,
				label="Vatican II ends (1965)")
axes[1].axvline(1891, color="orange", linestyle="--", linewidth=1.2,
				alpha=0.8, label="Rerum Novarum (1891)")
axes[1].set_xlabel("Year")
axes[1].set_ylabel("VADER Compound Score")
axes[1].set_title("Sentiment Over Time")
axes[1].legend(fontsize=9)
axes[1].grid(alpha=0.3)

plt.suptitle("VADER Sentiment Analysis", fontweight="bold", fontsize=14)
plt.tight_layout()
fig.savefig(FIGURES / "fig04_sentiment.png")
plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Figure 5: PCA scree
# ─────────────────────────────────────────────────────────────────────────────
print("Fig 5: PCA scree...")
cumvar = explained_var.cumsum()
fig, ax = plt.subplots(figsize=(9, 5))
ax.bar(range(len(explained_var)), explained_var.values * 100,
	   color="steelblue", alpha=0.75, label="Per component")
ax.plot(range(len(explained_var)), cumvar.values * 100,
		marker="o", color="darkred", linewidth=2, markersize=6, label="Cumulative")
ax.set_xticks(range(len(explained_var)))
ax.set_xticklabels([f"PC{i}" for i in range(len(explained_var))], rotation=45)
ax.set_xlabel("Principal Component")
ax.set_ylabel("Variance Explained (%)")
ax.set_title("PCA Scree Plot -- TF-IDF Document-Term Matrix")
ax.legend()
ax.grid(axis="y", alpha=0.4)
plt.tight_layout()
fig.savefig(FIGURES / "fig05_pca_scree.png")
plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Figure 6: PCA scatter by Vatican II period
# ─────────────────────────────────────────────────────────────────────────────
print("Fig 6: PCA Vatican II...")
period_colors = {
	"Pre-Vatican II (< 1962)":       "#d32f2f",
	"Vatican II transition (1962-65)": "#ff9800",
	"Post-Vatican II (> 1965)":       "#1565c0",
	"Date unknown":                   "#9e9e9e",
}


def pca_period(y):
	if pd.isna(y):  return "Date unknown"
	if y < 1962:    return "Pre-Vatican II (< 1962)"
	if y <= 1965:   return "Vatican II transition (1962-65)"
	return "Post-Vatican II (> 1965)"


pca_vii = DOC_PCA.join(LIBRARY[["year_num"]])
pca_vii["period"] = pca_vii["year_num"].apply(pca_period)

fig, axes = plt.subplots(1, 2, figsize=(18, 7))
for ax, (pcx, pcy) in zip(axes, [("PC0", "PC1"), ("PC1", "PC2")]):
	for period, color in period_colors.items():
		mask = pca_vii["period"] == period
		n = mask.sum()
		ax.scatter(pca_vii.loc[mask, pcx], pca_vii.loc[mask, pcy],
				   label=f"{period} (n={n})", s=60, alpha=0.55,
				   color=color, edgecolors="none")
	ax.set_xlabel(f"{pcx} ({explained_var[pcx]:.2%} var.)")
	ax.set_ylabel(f"{pcy} ({explained_var[pcy]:.2%} var.)")
	ax.set_title(f"PCA: {pcx} vs {pcy}")
	ax.legend(fontsize=9)
	ax.grid(alpha=0.3)
axes[0].text(0.02, 0.98,
			 "<-- English dominant      Italian/Latin dominant -->",
			 transform=axes[0].transAxes, fontsize=8, va="top",
			 color="gray", style="italic")
plt.suptitle("RQ3: Vatican II as Linguistic Rupture -- PCA Document Space",
			 fontweight="bold", fontsize=14)
plt.tight_layout()
fig.savefig(FIGURES / "fig06_pca_vatican2.png")
plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Figure 7: PCA scatter coloured by pope (top 9, chronological)
# ─────────────────────────────────────────────────────────────────────────────
print("Fig 7: PCA by pope...")
palette9   = sns.color_palette("tab10", n_colors=len(top9_chron))
pope_cmap  = dict(zip(top9_chron, palette9))
pca_meta   = DOC_PCA.join(LIBRARY[["pope", "year_num"]])
pca_top    = pca_meta[pca_meta["pope"].isin(top9_chron)]

fig, axes = plt.subplots(1, 2, figsize=(18, 7))
for ax, (pcx, pcy) in zip(axes, [("PC0", "PC1"), ("PC1", "PC2")]):
	for pope in top9_chron:
		mask = pca_top["pope"] == pope
		ax.scatter(pca_top.loc[mask, pcx], pca_top.loc[mask, pcy],
				   label=short_name(pope), s=70, alpha=0.65,
				   color=pope_cmap[pope],
				   edgecolors="white", linewidth=0.3)
	ax.set_xlabel(f"{pcx} ({explained_var[pcx]:.2%} var.)")
	ax.set_ylabel(f"{pcy} ({explained_var[pcy]:.2%} var.)")
	ax.set_title(f"PCA: {pcx} vs {pcy} -- Top 9 Popes (chronological legend)")
	ax.legend(fontsize=8, loc="upper right")
	ax.grid(alpha=0.3)
plt.suptitle("RQ2: Papal Authorial Signatures in PCA Space",
			 fontweight="bold", fontsize=14)
plt.tight_layout()
fig.savefig(FIGURES / "fig07_pca_pope.png")
plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Figure 8: PCA term loadings (PC0, PC1, PC2)
# ─────────────────────────────────────────────────────────────────────────────
print("Fig 8: PCA loadings...")
pc_interp = {
	"PC0": "Italian/Latin vs. English",
	"PC1": "Latin ecclesiastical vs. Modern English",
	"PC2": "Social/development vs. Classical hierarchical",
}
fig, axes = plt.subplots(1, 3, figsize=(18, 7))
for ax, pc in zip(axes, ["PC0", "PC1", "PC2"]):
	top_pos  = LOADINGS[pc].nlargest(12)
	top_neg  = LOADINGS[pc].nsmallest(12)
	combined = pd.concat([top_neg, top_pos]).sort_values()
	bcolors  = ["#c62828" if v < 0 else "#1565c0" for v in combined.values]
	combined.plot(kind="barh", ax=ax, color=bcolors, alpha=0.85)
	ax.set_title(f"{pc} Loadings\n({pc_interp[pc]})",
				 fontweight="bold", fontsize=11)
	ax.set_xlabel("Loading")
	ax.axvline(0, color="gray", linewidth=0.8)
	ax.grid(axis="x", alpha=0.3)
plt.suptitle("PCA Term Loadings -- Vocabulary Drivers of Document Separation",
			 fontweight="bold", fontsize=13)
plt.tight_layout()
fig.savefig(FIGURES / "fig08_pca_loadings.png")
plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Figure 9: LDA topic-pope heatmap (all popes >= 2 docs, chronological)
# ─────────────────────────────────────────────────────────────────────────────
print("Fig 9: LDA heatmap...")
topics_with_pope = DOC_TOPICS.join(LIBRARY[["pope"]])
pope_topics = (topics_with_pope[topics_with_pope["pope"].isin(sig_popes_chron)]
			   .groupby("pope")[DOC_TOPICS.columns].mean())
pope_topics_chron = pope_topics.loc[
	[p for p in sig_popes_chron if p in pope_topics.index]]
yticklabels = [short_name(p) for p in pope_topics_chron.index]

fig, ax = plt.subplots(figsize=(17, max(8, len(yticklabels) * 0.5)))
col_names_display = list(pope_topics_chron.rename(columns=topic_short).columns)
sns.heatmap(
	pope_topics_chron.rename(columns=topic_short),
	annot=True, fmt=".2f", cmap="YlOrRd",
	ax=ax, linewidths=0.4,
	cbar_kws={"label": "Mean topic weight"},
	annot_kws={"size": 8},
	yticklabels=yticklabels,
)
ax.set_title("Mean LDA Topic Weight per Pope (chronological, top 3 terms per topic)",
			 fontweight="bold", fontsize=12)
ax.set_xlabel("LDA Topic (top 3 terms)")
ax.set_ylabel("")
ax.tick_params(axis="y", labelsize=9)
ts_rev = {v: k for k, v in topic_short.items()}
for i, cn in enumerate(col_names_display):
	if ts_rev.get(cn, "") in artifact_topics:
		ax.add_patch(plt.Rectangle(
			(i, 0), 1, len(yticklabels),
			fill=False, edgecolor="blue", linewidth=2, clip_on=False))
ax.text(1.02, -0.04, "Blue border = language artifact topic",
		transform=ax.transAxes, fontsize=8, color="blue", style="italic")
plt.tight_layout()
fig.savefig(FIGURES / "fig09_lda_heatmap.png")
plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Figure 10: LDA topic evolution by decade
# ─────────────────────────────────────────────────────────────────────────────
print("Fig 10: topic evolution...")
topics_time = DOC_TOPICS.join(LIBRARY[["year_num"]]).dropna(subset=["year_num"])
topics_time["decade"] = (topics_time["year_num"] // 10 * 10).astype(int)
decade_topics = topics_time.groupby("decade")[DOC_TOPICS.columns].mean()

fig, axes = plt.subplots(2, 5, figsize=(22, 9), sharex=True)
for ax, col in zip(axes.flat, list(DOC_TOPICS.columns)):
	top3     = TOPIC_TERMS[col].nlargest(3).index.tolist()
	is_art   = col in artifact_topics
	color    = "gray" if is_art else "steelblue"
	ax.scatter(decade_topics.index, decade_topics[col] * 100,
			   alpha=0.5, s=40, color=color)
	roll = decade_topics[col].rolling(3, center=True, min_periods=1).mean()
	ax.plot(decade_topics.index, roll * 100,
			color="darkred" if not is_art else "gray", linewidth=2)
	ax.axvline(1965, color="gray", linestyle=":", linewidth=1, alpha=0.7)
	title = ", ".join(top3) + ("\n(lang. artifact)" if is_art else "")
	ax.set_title(title, fontsize=8.5, fontweight="bold",
				 color="gray" if is_art else "black")
	ax.set_ylabel("Weight (%)", fontsize=8)
	ax.grid(alpha=0.3)
	ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f"))
for ax in axes[-1]:
	ax.set_xlabel("Decade", fontsize=9)
	ax.tick_params(axis="x", rotation=45)
fig.suptitle("LDA Topic Evolution by Decade  (dotted line = Vatican II, 1965)\n"
			 "Gray panels = language artifact topics (Italian/Latin)",
			 fontsize=13, fontweight="bold")
plt.tight_layout()
fig.savefig(FIGURES / "fig10_topic_evolution.png")
plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Figure 11: Vocabulary shift pre/post Vatican II
#   Filtered to English-dominant docs (lang_ok=True) to avoid Italian artifact
# ─────────────────────────────────────────────────────────────────────────────
print("Fig 11: vocab shift (English-dominant docs only)...")
eng_pre  = [d for d in pre_vii  if d in TFIDF_DTM.index
			and LIBRARY.loc[d, "lang_ok"]]
eng_post = [d for d in post_vii if d in TFIDF_DTM.index
			and LIBRARY.loc[d, "lang_ok"]]

# Require term to appear in at least 20 documents (filters scraping artifacts
# which appear in only a few pages' nav bars despite high TF-IDF there)
min_df = 20
alpha_terms = [
	c for c in TFIDF_DTM.columns
	if c.isalpha() and len(c) > 2
	and c in VOCAB.index and VOCAB.loc[c, "df"] >= min_df
]
pre_mean  = TFIDF_DTM.loc[eng_pre,  alpha_terms].mean().nlargest(15)
post_mean = TFIDF_DTM.loc[eng_post, alpha_terms].mean().nlargest(15)

print(f"  Pre-Vatican II top 15 terms (n={len(eng_pre)} docs):")
print(f"  {list(pre_mean.index)}")
print(f"  Post-Vatican II top 15 terms (n={len(eng_post)} docs):")
print(f"  {list(post_mean.index)}")

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
pre_mean.plot(kind="barh", ax=axes[0], color="#c62828", alpha=0.8)
axes[0].set_title(
	f"Pre-Vatican II (<=1961)\nTop 15 TF-IDF terms\n"
	f"(English-dominant docs, n={len(eng_pre)})", fontweight="bold")
axes[0].set_xlabel("Mean TF-IDF")
axes[0].invert_yaxis()
axes[0].grid(axis="x", alpha=0.4)

post_mean.plot(kind="barh", ax=axes[1], color="#1565c0", alpha=0.8)
axes[1].set_title(
	f"Post-Vatican II (>1965)\nTop 15 TF-IDF terms\n"
	f"(English-dominant docs, n={len(eng_post)})", fontweight="bold")
axes[1].set_xlabel("Mean TF-IDF")
axes[1].invert_yaxis()
axes[1].grid(axis="x", alpha=0.4)

plt.suptitle("RQ3: Vocabulary Shift Across Vatican II",
			 fontweight="bold", fontsize=14)
plt.tight_layout()
fig.savefig(FIGURES / "fig11_vocab_shift.png")
plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Figure 12: Dispersion of social-teaching terms
# ─────────────────────────────────────────────────────────────────────────────
print("Fig 12: dispersion plot...")
social_terms = ["labor", "justice", "poor", "dignity", "social", "capital",
				"wage", "worker", "poverty", "ecology", "environment",
				"rights", "solidarity", "freedom"]
available = [t for t in social_terms if t in TFIDF_DTM.columns]
doc_order = (LIBRARY.dropna(subset=["year_num"])
			 .sort_values("year_num").index.tolist())
doc_pos = {d: i for i, d in enumerate(doc_order)}

fig, ax = plt.subplots(figsize=(16, 8))
for ti, term in enumerate(available):
	docs_with = TFIDF_DTM.index[TFIDF_DTM[term] > 0]
	positions = [doc_pos[d] for d in docs_with if d in doc_pos]
	ax.scatter(positions, [ti] * len(positions),
			   marker="|", s=200, linewidths=2, alpha=0.7, color="steelblue")

first_post   = next((doc_pos[d] for d in doc_order
					 if LIBRARY.loc[d, "year_num"] > 1965), None)
first_social = next((doc_pos[d] for d in doc_order
					 if LIBRARY.loc[d, "year_num"] >= 1891), None)
if first_post:
	ax.axvline(first_post, color="darkred", linestyle="--", linewidth=1.5,
			   label="Vatican II ends (1965)")
if first_social:
	ax.axvline(first_social, color="orange", linestyle="--", linewidth=1.2,
			   alpha=0.8, label="Rerum Novarum (1891)")
ax.legend(fontsize=10)
ax.set_yticks(range(len(available)))
ax.set_yticklabels(available, fontsize=11)
ax.set_xlabel("Documents ordered by year", fontsize=11)
ax.set_title("RQ1: Dispersion of Social-Teaching Terms Across Corpus\n"
			 "(each mark = document containing the term)",
			 fontweight="bold", fontsize=12)
ax.grid(axis="x", alpha=0.2)
plt.tight_layout()
fig.savefig(FIGURES / "fig12_dispersion.png")
plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Figure 13: t-SNE of Word2Vec embeddings
# ─────────────────────────────────────────────────────────────────────────────
print("Fig 13: t-SNE (may take ~60s)...")
top_terms = VOCAB.nlargest(400, "n").index
top_terms = top_terms[top_terms.isin(EMBEDDINGS.index)]
emb_sub   = EMBEDDINGS.loc[top_terms].values
tsne      = TSNE(n_components=2, perplexity=35, random_state=42,
				 n_iter=1000, learning_rate="auto", init="pca")
emb_2d    = tsne.fit_transform(emb_sub)
term_list2 = list(top_terms)

fig, ax = plt.subplots(figsize=(14, 10))
ax.scatter(emb_2d[:, 0], emb_2d[:, 1], s=20, alpha=0.4, color="steelblue")
for term in ["pope", "church", "christ", "faith", "justice", "love", "peace",
			 "labor", "poor", "world", "god", "spirit", "moral", "social",
			 "truth", "grace", "bishop", "holy", "divine", "freedom",
			 "sin", "human", "nature", "dignity", "solidarity"]:
	if term in term_list2:
		idx = term_list2.index(term)
		ax.annotate(term, xy=emb_2d[idx], fontsize=9, fontweight="bold",
					color="darkred",
					bbox=dict(boxstyle="round,pad=0.2",
							  facecolor="white", alpha=0.65))
ax.set_title("Word2Vec Semantic Space: t-SNE Projection (top 400 terms)",
			 fontweight="bold", fontsize=12)
ax.set_xlabel("t-SNE Dim 1")
ax.set_ylabel("t-SNE Dim 2")
plt.tight_layout()
fig.savefig(FIGURES / "fig13_tsne.png")
plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Figure 14: Modern papal topic profiles
# ─────────────────────────────────────────────────────────────────────────────
print("Fig 14: modern papal topic heatmap...")
modern_topics = (DOC_TOPICS.join(LIBRARY[["pope"]])
				 .loc[lambda df: df["pope"].isin(modern_popes_chron)]
				 .groupby("pope")[DOC_TOPICS.columns].mean()
				 .loc[modern_popes_chron])

fig, ax = plt.subplots(figsize=(16, 5))
sns.heatmap(
	modern_topics.rename(columns=topic_short),
	annot=True, fmt=".2f", cmap="Blues",
	ax=ax, linewidths=0.4,
	cbar_kws={"label": "Mean topic weight"},
	annot_kws={"size": 9},
	yticklabels=[short_name(p) for p in modern_popes_chron],
)
ax.set_title("RQ2: Mean LDA Topic Weight for Modern Popes (chronological)\n"
			 "Topic top 3 terms shown as labels",
			 fontweight="bold", fontsize=12)
ax.set_xlabel("LDA Topic (top 3 terms)")
plt.tight_layout()
fig.savefig(FIGURES / "fig14_modern_papal_topics.png")
plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Figure 15: Modern popes in PCA space
# ─────────────────────────────────────────────────────────────────────────────
print("Fig 15: modern pope PCA...")
modern_palette = sns.color_palette("Set2", n_colors=len(modern_popes_chron))
modern_cmap    = dict(zip(modern_popes_chron, modern_palette))
pca_m = (DOC_PCA.join(LIBRARY[["pope", "year_num"]])
		 .loc[lambda df: df["pope"].isin(modern_popes_chron)])

fig, ax = plt.subplots(figsize=(11, 7))
for pope in modern_popes_chron:
	mask = pca_m["pope"] == pope
	if mask.any():
		ax.scatter(pca_m.loc[mask, "PC0"], pca_m.loc[mask, "PC1"],
				   label=short_name(pope), s=110, alpha=0.78,
				   color=modern_cmap[pope],
				   edgecolors="black", linewidth=0.5)
ax.set_xlabel(
	f"PC0 ({explained_var['PC0']:.2%} var.)  <-- English     Italian -->")
ax.set_ylabel(f"PC1 ({explained_var['PC1']:.2%} var.)")
ax.set_title("RQ2: Modern Popes (post-1958) in PCA Document Space",
			 fontweight="bold", fontsize=12)
ax.legend(fontsize=10)
ax.grid(alpha=0.3)
plt.tight_layout()
fig.savefig(FIGURES / "fig15_modern_pca.png")
plt.close()


# ─────────────────────────────────────────────────────────────────────────────
saved = sorted(FIGURES.glob("*.png"))
print(f"\nDone. {len(saved)} figures saved to {FIGURES}.")
for p in saved:
	print(f"  {p.name}")
