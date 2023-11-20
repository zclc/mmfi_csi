import numpy as np

a = np.arange(0, 12).reshape([3, 4])

print(a)

b = a[:, [0, 2]]

print(b)
