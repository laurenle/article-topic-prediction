import json
import nltk
import numpy
from nltk.corpus import reuters

'''
Parse reuters corpus once so that subsequent calls to analyze.py
will not become bloated with slow connectivity/parsing latency.
'''

# there are lots of files in Reuters
limit_files_parsed = True
file_limit = 500

# get term frequency from corpus
term_freq = nltk.FreqDist()

# track file number
file_no = 0
doc_freq = {}

# retrieve data from reuters corpus
for fileid in reuters.fileids():
  if limit_files_parsed and file_no >= file_limit:
    break

  num_words = len(reuters.raw(fileid))
  file = reuters.words(fileid)

  for word in file:
    token = word.lower()
    # skip tokens that are numbers
    if token.isnumeric():
      continue
    if token in doc_freq and file_no in doc_freq[token]:
      doc_freq[token][file_no] += 1
    elif token in doc_freq:
      doc_freq[token][file_no] = 1
    else:
      doc_freq[token] = { file_no: 1 }

  # normalize each entry for document length
  for token in doc_freq:
    if file_no in doc_freq[token]:
      doc_freq[token][file_no] = float(doc_freq[token][file_no]) / float(num_words)
    else:
      doc_freq[token][file_no] = 0.0

  file_no += 1

for token in doc_freq:
  term_freq[token] = [numpy.mean(doc_freq[token].values())]

with open('term-freq.txt', 'w') as outfile:
  json.dump(term_freq, outfile)