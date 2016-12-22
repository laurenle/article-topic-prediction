import numpy as np
from math import sqrt
from collections import defaultdict

'''
Given two TF-IDF vectors, make both contain the same values
'''
def pad_missing_values(X, Y):
  X_padded = X.copy()
  Y_padded = Y.copy()
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
  # vectorize x and y
  padded_vectors = pad_missing_values(X, Y)
  vx = padded_vectors[0]
  vy = padded_vectors[1]
  sum_vxvy = 0.0
  sum_vx2 = 0.0
  for token in vx:
    sum_vx2 += vx[token] ** 2
    sum_vxvy += vx[token] * vy[token]
  sum_vy2 = 0.0
  for token in vy:
    sum_vy2 += vy[token] ** 2
  similarity = sum_vxvy / (sqrt(sum_vx2) * sqrt(sum_vy2))
  return similarity

'''
Given a matrix of values, print it appropriately
'''
def print_matrix(A):
  n = len(A)
  for row in range(n):
    print str(A[row]).strip('[]')