import json
import nltk
import numpy
from nltk.corpus import reuters
from collections import defaultdict
from text_analysis import sanitize_tokens

'''
Parse reuters corpus once so that subsequent calls to analyze.py
will not become bloated with slow connectivity/parsing latency.
'''

# there are lots of files in Reuters
limit_files_parsed = True
file_limit = 1000
idf_filename = 'doc-freq.txt'

# track file number
file_no = 0
df = defaultdict(int)
idf = defaultdict(float)

# retrieve data from reuters corpus
for fileid in reuters.fileids():
  if limit_files_parsed and file_no >= file_limit:
    break

  file = reuters.raw(fileid)
  words = nltk.word_tokenize(file)
  sanitized_words = sanitize_tokens(words)
  word_freq = nltk.FreqDist(sanitized_words)
  for token in word_freq:
    # only increment document frequency once for each file
    df[token] += 1
  file_no += 1

# store the number of documents analyzed and the idf vector
idf_data = (file_no, df)

with open(idf_filename, 'w') as outfile:
  json.dump(idf_data, outfile)