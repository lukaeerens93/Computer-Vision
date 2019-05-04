import numpy as np
import scipy.io
from nn import *
import matplotlib.pyplot as plt
from skimage.measure import compare_psnr as psnr

train_data = scipy.io.loadmat('../data/nist36_train.mat')
valid_data = scipy.io.loadmat('../data/nist36_valid.mat')

# we don't need labels now!
train_x = train_data['train_data']
valid_x = valid_data['valid_data']

dim = 32
# do PCA
dot_prod = np.dot(train_x.T, train_x)
U, S, Vh = np.linalg.svd(dot_prod)

# rebuild a low-rank version
lrank = np.dot( valid_x, U[: ,:dim] )
# rebuild it
recon = np.dot(lrank, U[: ,:dim].T)

increment = 800
for i in range(5):
    plt.subplot(2,1,1)
    plt.imshow(valid_x[i+increment].reshape(32,32).T)
    plt.subplot(2,1,2)
    plt.imshow(recon[i+increment].reshape(32,32).T)
    plt.show()

# build valid dataset
recon_valid = np.dot(lrank, U[: ,:dim].T)

total = []
for pred,gt in zip(recon_valid,valid_x):
    total.append(psnr(gt,pred))
print(np.array(total).mean())