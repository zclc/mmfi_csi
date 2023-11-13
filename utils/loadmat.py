import scipy.io as sio
import os

root = 'E:\mmfids\E01\S01\A01\wifi-csi'

csiFrame = sio.loadmat(os.path.join(root, 'frame001.mat'))

print(csiFrame)
