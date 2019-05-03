import numpy as np
from q2 import *
from q3 import *
import skimage.color

# you may find this useful in removing borders
# from pnc series images (which are RGBA)
# and have boundary regions
def clip_alpha(img):
    img[:,:,3][np.where(img[:,:,3] < 1.0)] = 0.0
    return img 

# Q 4.1
# this should be a less hacky version of
# composite image from Q3
# img1 and img2 are RGB-A
# warp order 0 might help
# try warping both images then combining
# feel free to hardcode output size
def imageStitching(img1, img2, H2to1):
    panoImg = None
    thresh = 4      # Variable
    w = np.linalg.inv( H2to1 / H2to1[2,2] )
    s = (img1.shape[0],500,3)

    # Configure stitiching policies
    p_img = warp(img2, w, output_shape = s)
    shape_of_pano = panoImg.s
 
    if (len(shape_of_pano) == thresh):
        w_img = clip_alpha( warp(img1, np.eye(3), output_shape = s) )
        p_img = clip_alpha( warp(img2, w, output_shape = s) )
        
    plt.imshow(p_img)
    plt.show()
    
    
    return panoImg


# Q 4.2
# you should make the whole image fit in that width
# python may be inv(T) compared to MATLAB
def imageStitching_noClip(img1, img2, H2to1, panoWidth=1280):
    panoImg = None
    # YOUR CODE HERE
    
    return panoImg

# Q 4.3
# should return a stitched image
# if x & y get flipped, np.flip(_,1) can help
def generatePanorama(img1, img2):
    panoImage = None
    # YOUR CODE HERE
    
    return panoImage

# Q 4.5
# I found it easier to just write a new function
# pairwise stitching from right to left worked for me!
def generateMultiPanorama(imgs):
    panoImage = None
    # YOUR CODE HERE
    
    return panoImage
