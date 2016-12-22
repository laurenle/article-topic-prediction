import sys
import codecs
import nltk
import os.path
import json
from collections import defaultdict
from vector_operations import *
from text_analysis import *

df_filename = 'doc-freq.txt'
input_dir = 'training-data'
file_no = 0
tf = {}
tf_idf = {}
cap_files_analyzed = False
max_files = 6

# retrieve pre-processed term frequency data
with open(df_filename) as json_data:
    df_data = json.load(json_data)
    json_data.close()

# extract idf vector and number of documents analyzed from JSON
docs_analyzed = df_data[0]
df = df_data[1]

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
    tf[file_no] = get_tf(words)
    # use words parsed to update idf vector
    df = update_df(df, words)
    # store one tf_idf vector for every story analyzed
    file_no += 1

for i in range(file_no):
    tf_idf[i] = get_tf_idf(tf[i], df, docs_analyzed + file_no)

similarity = get_similarity_matrix(tf_idf)
print_matrix(similarity)

