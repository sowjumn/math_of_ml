import numpy as np

A = [[1, 1, 2],
     [2, 5, 6],
     [3, 1, 0]]

Ainv = np.linalg.inv(A)

print(Ainv)