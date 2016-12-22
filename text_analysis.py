import nltk
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from collections import defaultdict
import numpy
import math
from vector_operations import *

max_words_returned = 20

'''
Given a list of words, output the term frequency
'''
def get_tf(words):
  sanitized_words = sanitize_tokens(words)
  word_freq = nltk.FreqDist(sanitized_words)
  tf = {}
  for word, freq in word_freq.most_common(max_words_returned):
      if word is None:
        continue
      tf[word] = float(freq) / float(len(words))
  return tf

'''
Given an integer representing # of occurrences for a token
and an integer representing # of documents analyzed, 
return a float representing the token's idf value
'''
def compute_idf(occurrences, docs_analyzed):
  return math.log(docs_analyzed / occurrences)

'''
Given a dict of term frequencies and a dict of 
inverse document frequencies, output a dict of
tf-idf values.
'''
def get_tf_idf(tf, df, docs_analyzed):
  tf_idf = {}
  for word in tf:
    if word in df:
      tf_idf[word] = tf[word] * compute_idf(df[word], docs_analyzed)
  return tf_idf

'''
Given a matrix, compute the cosine similarity
for every pair of vectors and return as a matrix.
'''
def get_similarity_matrix(A):
  # Define matrix size
  n = len(A)
  similarity = [[0.0 for x in range(n)] for y in range(n)]
  for v1 in range(n):
    for v2 in range(n):
      if v1 == v2:
        similarity[v1][v2] = 1.0
      else:
        similarity[v1][v2] = round(cosine_similarity(A[v1], A[v2]), 1)
  return similarity

'''
Given a list of words, sanitize the list to omit data that
is not useful for our analysis
'''
def sanitize_tokens(words):
  all_stopwords = set(nltk.corpus.stopwords.words('english'))
  # place all words in lowercase
  words = [word.lower() for word in words]
  # Strip stopwords and numerics
  words = [word for word in words if not word.isnumeric() 
      and word not in all_stopwords]
  # remove single-character tokens (mostly punctuation)
  words = [word for word in words if len(word) > 2]
  # reduce each word to morphological root
  words = [wn.morphy(word) for word in words]
  return words

'''
Given a df vector, the previous number of documents analyzed,
and a list of words for a new document, integrate new words with
df
'''
def update_df(df, words):
  sanitized_words = sanitize_tokens(words)
  word_freq = nltk.FreqDist(sanitized_words)
  updated_df = defaultdict(float)
  updated_df = df.copy()
  for token in word_freq:
    if token in updated_df:
      updated_df[token] += 1
    else:
      updated_df[token] = 1
  return updated_df


