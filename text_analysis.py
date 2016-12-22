import nltk
from nltk.corpus import stopwords

max_words_returned = 30

'''
Given a list of words, output the term frequency
'''
def get_tf(words):
  sanitized_words = sanitize_tokens(words)
  word_freq = nltk.FreqDist(sanitized_words)
  tf = {}
  for word, freq in word_freq.most_common(max_words_returned):
      tf[word] = float(freq) / float(len(words))
  return tf

'''
Given a dict of term frequencies and a dict of 
inverse document frequencies, output a dict of
tf-idf values.
'''
def get_tf_idf(tf, idf):
  tf_idf = {}
  for word in tf:
    if word in idf:
        tf_idf[word] = tf[word] * idf[word]
  return tf_idf

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
  return words