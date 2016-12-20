import sys
import codecs
import nltk
import os.path
from nltk.corpus import stopwords

all_stopwords = set(nltk.corpus.stopwords.words('english'))

input_dir = 'training-data'

for dir, subdirs, files in os.walk(input_dir):
  for file in files:
    # ignore hidden files
    if file[0] == '.':
        continue

    input_file = input_dir + "/" + file
    fp = codecs.open(input_file, 'r', 'UTF-8')
    words = nltk.word_tokenize(fp.read())

    # Numbers generally not helpful for analytics
    words = [word for word in words if not word.isnumeric()]

    # place all words in lowercase
    words = [word.lower() for word in words]

    # remove single-character tokens (mostly punctuation)
    words = [word for word in words if len(word) > 1]

    # remove stopwords. again, stopwords may or may not improve analytics
    words = [word for word in words if word not in all_stopwords]

    # use nltk to calculate document frequency
    df = nltk.FreqDist(words)

    # Output top 50 words
    for word, frequency in df.most_common(50):
        print(u'{};{}'.format(word, frequency,))
