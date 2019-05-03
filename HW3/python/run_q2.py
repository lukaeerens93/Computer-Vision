from q2 import makeTestPattern, key_point_detector, computeBrief, briefLite, briefMatch, testMatch, briefRotTest, briefRotLite

import scipy.io as sio
import skimage.color
import skimage.io
import skimage.feature as feat
import matplotlib.pyplot as plt
import time

# Q2.1
compareX, compareY, uX, uY = makeTestPattern(9,256)
sio.savemat('testPattern.mat',{'compareX':compareX,'compareY':compareY})

# Q2.2
img = skimage.io.imread('../data/chickenbroth_01.jpg')
im = skimage.color.rgb2gray(img)

# YOUR CODE: Run a keypoint detector, with nonmaximum supression
# locs holds those locations n x 2
locs = None

# Find Harris Corner Peaks
locs = key_point_detector(locs, compareX, im)

locs, desc = computeBrief(im,locs,compareX,compareY, uX, uY)

# Q2.3
locs, desc = briefLite(im, compareX, compareY, uX, uY)

# Q2.4
testMatch(compareX, compareY, uX, uY)

# Q2.5
briefRotTest(compareX, compareY, uX, uY, briefRotLite)

# EC 1
briefRotTest(briefRotLite)

# EC 2 
# write it yourself!


