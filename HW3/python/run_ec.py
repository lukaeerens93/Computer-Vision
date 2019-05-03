import skimage.color
import skimage.io
import numpy as np
import matplotlib.pyplot as plt
import math
from ec import *
#foreground
fg = skimage.io.imread('../data/fg2.png')[:,:,:3]

# background
bg = skimage.io.imread('../data/bg2.png')[:,:,:3]

# binary mask should be grey
mask = skimage.io.imread('../data/mask2.png')[:,:,:3]
mask[:,:,1] = mask[:,:,0] 
mask[:,:,2] = mask[:,:,0]  

# bad clone
cloned = np.copy(bg)
cloned[np.where(mask >0)] = fg[np.where(mask>0)]

plt.subplot(1,2,1)
plt.imshow(cloned)
plt.title('before')
seamless = poissonStitch(fg,bg,mask)
plt.subplot(1,2,2)
plt.imshow(seamless)
plt.title('after')
plt.show()