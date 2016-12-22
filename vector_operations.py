import numpy as np
from math import sqrt
from collections import defaultdict

'''
Given two TF-IDF vectors, make both contain the same values
'''
def pad_missing_values(X, Y):
  X_padded = X
  Y_padded = Y
  for token in X:
    if token not in Y:
      Y_padded[token] = 0.0
  for token in Y:
    if token not in X:
      X_padded[token] = 0.0
  return (X_padded, Y_padded)

'''
Compute the cosine similarity of two TF-IDF vectors
'''
def cosine_similarity(X, Y):
  padded_vectors = pad_missing_values(X, Y)
  # vectorize x and y
  vx = padded_vectors[0]
  vy = padded_vectors[1]
  sum_vxvy = 0.0
  sum_vx2 = 0.0
  for token in X:
    sum_vx2 += X[token] ** 2
    sum_vxvy += X[token] * Y[token]
  sum_vy2 = 0.0
  for token in Y:
    sum_vy2 += Y[token] ** 2
  similarity = sum_vxvy / (sqrt(sum_vx2) * sqrt(sum_vy2))
  return similarity

'''
Given a matrix of values, print it appropriately
'''
def print_matrix(A):
  # TODO
  return