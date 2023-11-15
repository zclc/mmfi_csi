import numpy as np
import matplotlib.pyplot as plt
import os

stick_defines = [
    (0, 1), (1, 2), (2, 3), (3, 26), (26, 27), (27, 28), (27, 29), (27, 30),
    (27, 31),
    (2, 11), (11, 12), (12, 13), (13, 14),
    (2, 4), (4, 5), (5, 6), (6, 7),
    (0, 18), (18, 19), (19, 20), (20, 21),
    (0, 22), (22, 23), (23, 24), (24, 25)
]

root = '/media/zcl/6EC1C7A5C00D55DD/mmfids/E01/S01/A01/ground_truth.npy'

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

a = np.load(root)
ax.plot(a[:, 0], a[:, 1], a[:, 2])
ax.show()
