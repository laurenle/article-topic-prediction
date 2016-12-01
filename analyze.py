import sys
import codecs
import nltk
from nltk.corpus import stopwords
from nltk.corpus import reuters

# get term frequency from corpus
tf = nltk.FreqDist()

# retrieve data from reuters corpus
for sentence in reuters.sents():
  for word in sentence:
    tf[word.lower()] += 1

all_stopwords = set(nltk.corpus.stopwords.words('english'))

input_file = 'articles/2016-11-30-nyt-north-carolinas-sore-loser.txt'

fp = codecs.open(input_file, 'r', 'utf-8')

words = nltk.word_tokenize(fp.read())

# Numbers may or may not be valuable for analytics
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
    print(u'{};{};{}'.format(word, frequency, tf[word]))