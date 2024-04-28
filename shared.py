from env import env

import pandas as pd
import pickle
import math
from whoosh.analysis import Filter                                
from nltk.stem import WordNetLemmatizer

# load the wiki dataframe
wiki_df = None
def get_wiki_df():
    global wiki_df
    if wiki_df is None:
        wiki_df = pd.read_pickle(f"{env.data_dir}/wiki.pkl")
    return wiki_df

# load the question dataframe
questions_df = None
def get_questions_df():
    global questions_df
    if questions_df is None:
        questions_df = pd.read_pickle(f"{env.data_dir}/questions.pkl")
    return questions_df

# create the lemmatize filter
lemmatizer = WordNetLemmatizer()
class LemmatizeFilter(Filter):
    def __call__(self, tokens):
        for token in tokens:
            token.text = lemmatizer.lemmatize(token.text)
            yield token

# define the query filter
with open(f"{env.data_dir}/term_counts.pkl", "rb") as file:
    term_counts = pickle.load(file)
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

# transform the category by boosting each term in the category by 0.5
def transform_category(category):
    new_cat = ""
    for c in category.split():
        new_cat += c + "^0.5 "
    return new_cat.strip()