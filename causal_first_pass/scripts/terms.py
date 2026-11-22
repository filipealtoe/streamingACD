import pandas as pd, re, json, numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
OUT = r"C:\StreamingACD\causal_first_pass"
texts = pd.read_parquet(OUT + r"\narrative_texts_first10.parquet")
def clean(x):
    x = re.sub(r"https?://\S+|@\w+|&amp;|\bRT\b", " ", x)
    return re.sub(r"#", "", x)
docs = texts.narrative_text.map(clean)
vec = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", min_df=2, max_df=0.2,
                      token_pattern=r"(?u)\b[A-Za-z][A-Za-z']+\b", sublinear_tf=True)
X = vec.fit_transform(docs); vocab = np.array(vec.get_feature_names_out())
terms = set()
for i in range(X.shape[0]):
    row = X[i].toarray().ravel(); top = row.argsort()[::-1][:6]
    bi = [vocab[j] for j in top if " " in vocab[j] and row[j] > 0][:2]
    uni = [vocab[j] for j in top if " " not in vocab[j] and row[j] > 0][:1]
    terms.update(bi + uni)
seeds = ["ballots", "mail-in ballots", "voter fraud", "election fraud", "dominion", "poll watchers", "dead voters",
         "biden", "trump", "kamala harris", "covid", "coronavirus", "mask", "vaccine", "hunter biden", "laptop",
         "antifa", "black lives matter", "riots", "supreme court", "amy coney barrett", "debate", "obamacare",
         "social security", "fracking", "police", "defund", "china", "rigged", "sharpie", "stop the steal",
         "georgia", "pennsylvania", "michigan", "wisconsin", "arizona", "nevada", "postal service", "usps",
         "hydroxychloroquine", "fauci", "lockdown", "taxes", "medicare", "immigration", "qanon", "soros"]
terms = sorted(terms | set(seeds))
json.dump(terms, open(OUT + r"\factchecks\google_query_terms.json", "w"), indent=0)
print(len(terms), terms[:15])
