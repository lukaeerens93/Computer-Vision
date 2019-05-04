from q2 import eightpoint, sevenpoint
from q3 import essentialMatrix, triangulate
from util import plot_epipolar_lines, camera2

import sys
import os
import numpy as np
import scipy.io
import skimage.io
import matplotlib.pyplot as plt

# Setup
# can take homework_dir as first argument
HOMEWORK_DIR = ".." if len(sys.argv) < 2 else sys.argv[1]
SOME_CORRS = os.path.join(HOMEWORK_DIR,'data','some_corresp.mat')
INTRINS = os.path.join(HOMEWORK_DIR,'data','intrinsics.mat')

IM1_PATH = os.path.join(HOMEWORK_DIR,'data','im1.png')
IM2_PATH = os.path.join(HOMEWORK_DIR,'data','im2.png')

im1 = skimage.io.imread(IM1_PATH)
im2 = skimage.io.imread(IM2_PATH)
np.set_printoptions(suppress=True)

# Q3.1
corr = scipy.io.loadmat(SOME_CORRS)
pts1 = corr['pts1']
pts2 = corr['pts2']
#intrin = scipy.io.loadmat(INTRINS) #we wish
import h5py
with h5py.File(INTRINS, 'r') as f:
    K1 = np.array(f['K1']).T
    K2 = np.array(f['K2']).T

F = eightpoint(pts1,pts2,max(im1.shape))
F = F/F[2,2]

E = essentialMatrix(F,K1,K2)
E = E/E[2,2]
M2s = camera2(E)
print ('M2s' + str(M2s))
print ('K1' + str(K1))
print ('K2' + str(K2))
print ('F' + str(E))

# Q3.2 / 3.3
C1 = np.hstack([np.eye(3),np.zeros((3,1))])
for C2 in M2s:
    P, err = triangulate(K1.dot(C1),pts1,K2.dot(C2),pts2)
    if(P.min(0)[2] > 0):
        # we're the right one!
        c = np.dot(K2,C2)
        scipy.io.savemat('q3_3.mat', {'M2':C2,'C2':c,'p1':pts1,'p2':pts2,'P':P} )
        break
print(P,err)
print (C2)
print (c)