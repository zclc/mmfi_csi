import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mmfi_lib.evaluate import calulate_error, cal_mpjpe

root = r"E:\mmfids\E01\S01\A03\ground_truth.npy"

gts = np.load(root)

fig = plt.figure()
ax = fig.add_subplot(121, projection='3d')
plt.rcParams["font.sans-serif"] = ["SimHei"]  # 设置字体
plt.rcParams["axes.unicode_minus"] = False  # 该语句解决图像中的“-”负号的乱码问题

stick_defines = [
    (0, 1), (1, 2), (2, 3)
    , (0, 4), (4, 5), (5, 6)
    , (0, 7), (7, 8), (8, 9), (9, 10)
    , (8, 11), (11, 12), (12, 13)
    , (8, 14), (14, 15), (15, 16)
]

a = np.multiply(gts, 1000)

ax.scatter(a[0, :, 0], a[0, :, 1], a[0, :, 2], marker='o')

# for j1, j2 in stick_defines:
#     ax.plot(a[0, [j1, j2], 0], a[0, [j1, j2], 1], a[0, [j1, j2], 2], color='blue', marker='o')

ax2 = fig.add_subplot(122, projection='3d')
for j1, j2 in stick_defines:
    ax2.scatter(a[0, :, 0], a[0, :, 1], a[0, :, 2], marker='o')
    ax2.plot(a[0, [j1, j2], 0], a[0, [j1, j2], 1], a[0, [j1, j2], 2], color='blue', marker='o')

root2 = r"E:\mmfids\E01\S04\A02\ground_truth.npy"
preds = np.load(root2)
mpjpe, pmpjpe = calulate_error(np.multiply(preds, 1000), np.multiply(gts, 1000))

print(mpjpe, pmpjpe)

plt.show()
