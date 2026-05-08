"""Extract all verified stats for report/notebook rewrite."""
import pandas as pd, numpy as np, json
from pathlib import Path
from scipy.spatial.distance import cosine as cos_dist

P = Path('data/processed')
LIB  = pd.read_csv(P/'LIBRARY.csv',       index_col='doc_id')
VOC  = pd.read_csv(P/'VOCAB.csv',         index_col=0)
TT   = pd.read_csv(P/'TOPIC_TERMS.csv',   index_col=0)
DT   = pd.read_csv(P/'DOC_TOPICS.csv',    index_col='doc_id')
PCA  = pd.read_csv(P/'DOC_PCA.csv',       index_col='doc_id')
LOAD = pd.read_csv(P/'LOADINGS.csv',      index_col=0)
DTM  = pd.read_csv(P/'TFIDF_DTM.csv',     index_col='doc_id')
SENT = pd.read_csv(P/'DOC_SENTIMENT.csv', index_col='doc_id')
EV   = pd.read_csv(P/'explained_variance.csv', header=None)
EMBS = pd.read_csv(P/'EMBEDDINGS.csv',    index_col=0)

LIB['year_num'] = pd.to_numeric(LIB['year'], errors='coerce')

# ── Chronological pope order ──────────────────────────────────────────────────
pope_first_year = LIB.groupby('pope')['year_num'].min().sort_values()
print("=== Pope chronological order (by first document year) ===")
for p, y in pope_first_year.items():
    n = (LIB['pope']==p).sum()
    yr = int(y) if not np.isnan(y) else 'UNK'
    print(f"  {yr:>4}  n={n:>3}  {p}")

# ── Eras ──────────────────────────────────────────────────────────────────────
pre_vii  = LIB[LIB['year_num'] < 1962].index
trans    = LIB[(LIB['year_num'] >= 1962) & (LIB['year_num'] <= 1965)].index
post_vii = LIB[LIB['year_num'] > 1965].index
print(f"\npre-VII n={len(pre_vii)}, transition n={len(trans)}, post-VII n={len(post_vii)}, unknown n={(LIB['year_num'].isna()).sum()}")

# ── Sentiment ─────────────────────────────────────────────────────────────────
sc = 'sentiment_compound'
sp = 'sentiment_pos'
sn = 'sentiment_neg'
print(f"\n=== Sentiment ===")
print(f"Overall: mean={LIB[sc].mean():.3f}, median={LIB[sc].median():.3f}, std={LIB[sc].std():.3f}")
print(f"Pre-VII:  mean={LIB.loc[pre_vii, sc].mean():.3f}")
print(f"Trans:    mean={LIB.loc[trans, sc].mean():.3f}")
print(f"Post-VII: mean={LIB.loc[post_vii, sc].mean():.3f}")

modern_chron = ['Pope St. John XXIII','Pope Paul VI','Pope St. John Paul II','Pope Benedict XVI','Pope Francis']
print("\nModern popes sentiment (compound / pos / neg):")
for p in modern_chron:
    m = LIB['pope']==p
    if m.any():
        print(f"  {p}: {LIB.loc[m,sc].mean():.3f} / {LIB.loc[m,sp].mean():.3f} / {LIB.loc[m,sn].mean():.3f}  (n={m.sum()})")

# ── Topic labels ──────────────────────────────────────────────────────────────
print("\n=== LDA topic top-8 terms ===")
for col in TT.columns:
    print(f"  {col}: {', '.join(TT[col].nlargest(8).index.tolist())}")

# ── Topic means by pope (chronological, >=5 docs) ────────────────────────────
sig_popes = LIB['pope'].value_counts()
sig_popes = sig_popes[sig_popes>=5].index
merged = DT.join(LIB[['pope']])
pope_topics = merged[merged['pope'].isin(sig_popes)].groupby('pope')[DT.columns].mean()
chron = [p for p in pope_first_year.index if p in pope_topics.index]
print("\n=== Pope topic means (chronological) ===")
print(pope_topics.loc[chron].round(3).to_string())

print("\nDominant topic per pope:")
for p in chron:
    row = pope_topics.loc[p]
    dom = row.idxmax()
    top4 = ', '.join(TT[dom].nlargest(4).index.tolist())
    print(f"  {p:40s} {dom} ({row[dom]:.3f})  [{top4}]")

# ── Topic means by era ────────────────────────────────────────────────────────
def era_label(y):
    if pd.isna(y): return 'Unknown'
    if y < 1891:   return 'Pre-Modern (≤1890)'
    if y < 1962:   return 'Social Teaching (1891-1961)'
    if y <= 1965:  return 'Vatican II (1962-65)'
    if y <= 2000:  return 'Post-Vatican II (1966-2000)'
    return 'Contemporary (2001+)'

LIB['era_label'] = LIB['year_num'].apply(era_label)
merged2 = DT.join(LIB[['era_label']])
era_topics = merged2.groupby('era_label')[DT.columns].mean()
era_order = ['Pre-Modern (≤1890)','Social Teaching (1891-1961)','Vatican II (1962-65)','Post-Vatican II (1966-2000)','Contemporary (2001+)']
print("\n=== Mean topic weight by era ===")
print(era_topics.loc[[e for e in era_order if e in era_topics.index]].round(3).to_string())

# ── PCA explained variance ────────────────────────────────────────────────────
ev_vals = EV.iloc[1:,1].astype(float).values
print("\n=== PCA explained variance ===")
cumv = 0
for i,v in enumerate(ev_vals):
    cumv += v
    print(f"  PC{i}: {v:.4%}  cumul={cumv:.4%}")

# ── PCA loadings ──────────────────────────────────────────────────────────────
print("\n=== PCA loadings top/bottom 8 ===")
for pc in LOAD.columns[:5]:
    top = LOAD[pc].nlargest(8).round(4).to_dict()
    bot = LOAD[pc].nsmallest(8).round(4).to_dict()
    print(f"  {pc} positive: {list(top.keys())}")
    print(f"  {pc} negative: {list(bot.keys())}")

# ── TF-IDF top terms pre/post ─────────────────────────────────────────────────
# Filter out non-alpha tokens for readability
def alpha_only(s): return pd.Series({k:v for k,v in s.items() if k.isalpha() and len(k)>2})
pre_m  = alpha_only(DTM.loc[DTM.index.isin(pre_vii)].mean().nlargest(50))
post_m = alpha_only(DTM.loc[DTM.index.isin(post_vii)].mean().nlargest(50))
print("\n=== Top 15 alpha TF-IDF terms pre-Vatican II ===")
print(pre_m.nlargest(15).round(5).to_string())
print("\n=== Top 15 alpha TF-IDF terms post-Vatican II ===")
print(post_m.nlargest(15).round(5).to_string())

# ── Word2Vec nearest neighbors ────────────────────────────────────────────────
def nn(word, n=6):
    if word not in EMBS.index: return "(not in vocab)"
    v = EMBS.loc[word].values
    sims = EMBS.apply(lambda r: 1 - cos_dist(v, r.values), axis=1)
    return ', '.join(sims.nlargest(n+1).iloc[1:].index.tolist())

print("\n=== Word2Vec nearest neighbors ===")
for w in ['labor','justice','dignity','poor','christ','faith','pope','church','freedom','world','social','modern']:
    print(f"  {w:12s}: {nn(w)}")

# ── Vocab top non-stop terms ──────────────────────────────────────────────────
print("\n=== Top 20 non-stop vocab terms ===")
if 'is_stop' in VOC.columns:
    top_v = VOC[~VOC['is_stop'] & VOC.index.str.isalpha()].nlargest(20,'n')[['n','df']]
else:
    top_v = VOC[VOC.index.str.isalpha()].nlargest(20,'n')[['n','df']]
print(top_v.to_string())

print("\nDone.")
