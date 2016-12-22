import sys
import codecs
import nltk
import os.path
import json
from collections import defaultdict
from vector_operations import *
from text_analysis import *

idf_filename = 'inverse-doc-freq.txt'
input_dir = 'training-data'
file_no = 0
tf_idf = {}

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
    # read in file as a list of tokens
    words = nltk.word_tokenize(fp.read())
    # compute the term frequency for the current file
    tf = get_tf(words)
    # store one tf_idf vector for every story analyzed
    tf_idf[file_no] = get_tf_idf(tf, idf)
    file_no += 1

similarity = get_similarity_matrix(tf_idf)
print_matrix(similarity)

