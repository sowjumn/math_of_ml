import numpy as np

# Find r for Ar = s

A = [[4, 6, 2],
    [3, 4, 1],
    [2, 8, 13]]

s = [9, 7, 2]

soln = np.linalg.solve(A,s)

print(soln)