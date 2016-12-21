import sys
import codecs
import nltk
import os.path
import json
from nltk.corpus import stopwords

# retrieve pre-processed term frequency data
with open('term-freq.txt') as json_data:
    tf = json.load(json_data)
    json_data.close()

all_stopwords = set(nltk.corpus.stopwords.words('english'))

input_dir = 'training-data'

for dir, subdirs, files in os.walk(input_dir):
  for file in files:
    # ignore hidden files
    if file[0] == '.':
        continue

    input_file = dir + "/" + file
    fp = codecs.open(input_file, 'r', 'UTF-8')
    words = nltk.word_tokenize(fp.read())

    # Numbers generally not helpful for analytics
    words = [word for word in words if not word.isnumeric()]

    # place all words in lowercase
    words = [word.lower() for word in words]

    # remove single-character tokens (mostly punctuation)
    words = [word for word in words if len(word) > 2]

    # remove stopwords. again, stopwords may or may not improve analytics
    words = [word for word in words if word not in all_stopwords]

    # use nltk to calculate document frequency
    df = nltk.FreqDist(words)



    print "-----------------------------------"
    print input_file

    # Output top 50 words
    for word, frequency in df.most_common(10):
        if word in tf:
            print(u'{};{};{}'.format(word, frequency, tf[word]))
