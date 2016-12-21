import sys
import codecs
import nltk
import os.path
import json
from nltk.corpus import stopwords
from collections import defaultdict

max_words_returned = 20
idf_filename = 'inverse-doc-freq.txt'
input_dir = 'training-data'
all_stopwords = set(nltk.corpus.stopwords.words('english'))

# retrieve pre-processed term frequency data
with open(idf_filename) as json_data:
    idf = json.load(json_data)
    json_data.close()

for dir, subdirs, files in os.walk(input_dir):
  for file in files:
    # ignore hidden files
    if file[0] == '.':
        continue

    input_file = dir + "/" + file
    fp = codecs.open(input_file, 'r', 'UTF-8')
    words = nltk.word_tokenize(fp.read())

    # place all words in lowercase
    words = [word.lower() for word in words]

    # Strip stopwords and numerics
    words = [word for word in words if not word.isnumeric() 
        and word not in all_stopwords]

    # remove single-character tokens (mostly punctuation)
    words = [word for word in words if len(word) > 2]

    # use nltk to calculate document frequency
    word_freq = nltk.FreqDist(words)

    tf = defaultdict(float)
    for word, freq in word_freq.most_common(max_words_returned):
        tf[word] = float(freq) / float(len(words))

    print "-----------------------------------"
    print input_file

    for word in tf:
        if word in idf:
            print(u'{}; {}'.format(word, tf[word] * idf[word]))


