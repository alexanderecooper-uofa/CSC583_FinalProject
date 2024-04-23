from env import env

import pandas as pd
import pickle
import math
from whoosh.analysis import Filter                                
from nltk.stem import WordNetLemmatizer

wiki_df = pd.read_pickle(f"{env.data_dir}/wiki.pkl")
wiki_redirects_df = pd.read_pickle(f"{env.data_dir}/wiki_redirects.pkl")
questions_df = pd.read_pickle(f"{env.data_dir}/questions.pkl")

with open(f"{env.data_dir}/term_counts.pkl", "rb") as file:
    term_counts = pickle.load(file)

redirect_lookups = {}
for _, row in wiki_redirects_df.iterrows():
    if row.redirect_index in redirect_lookups:
        redirect_lookups[row.redirect_index].append(row.title)
    else:
        redirect_lookups[row.redirect_index] = [row.title]

lemmatizer = WordNetLemmatizer()

class LemmatizeFilter(Filter):
    def __call__(self, tokens):
        for token in tokens:
            token.text = lemmatizer.lemmatize(token.text)
            yield token

def get_term_count(term):
    if len(term) <= 1:
        return float("inf")
    if term.isnumeric():
        return 0
    if term in term_counts:
        return term_counts[term]
    else:
        return 0

def filter_query(query):
    query_subset_p = 0.75 # the percentage of the query to keep
    query = query.split()
    query_counts = sorted([(get_term_count(lemmatizer.lemmatize(t.lower())), i) for i, t in enumerate(query)])
    n = math.ceil(len(query) * query_subset_p)
    query_indices = sorted([i for _, i in query_counts[:n]])
    return " ".join([query[i] for i in query_indices])