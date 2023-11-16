import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mmfi_lib.evaluate import calulate_error, cal_mpjpe

root = r"E:\mmfids\E01\S01\A01\ground_truth.npy"

gts = np.load(root)

fig, ax = plt.subplots()  # a figure with a single Axes
ax = fig.add_subplot(111, projection='3d')


a = np.multiply(gts, 1000)
print(a[0, 1, :])
print(a[0, 2, :])

ax.scatter(a[0, :, 0], a[0, :, 1], a[0, :, 2])


root2 = r"E:\mmfids\E01\S04\A02\ground_truth.npy"
preds = np.load(root2)
mpjpe, pmpjpe = calulate_error(np.multiply(preds, 1000), np.multiply(gts, 1000))

print(mpjpe, pmpjpe)

plt.show()