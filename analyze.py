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
cap_files_analyzed = True
max_files = 6

# retrieve pre-processed term frequency data
with open(idf_filename) as json_data:
    idf_data = json.load(json_data)
    json_data.close()

# extract idf vector and number of documents analyzed from JSON
idf_docs_analyzed = idf_data[0]
idf = idf_data[1]

for dir, subdirs, files in os.walk(input_dir):
  for file in files:
    if cap_files_analyzed and file_no >= max_files:
        break
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
    print input_file
    file_no += 1

similarity = get_similarity_matrix(tf_idf)
print_matrix(similarity)

