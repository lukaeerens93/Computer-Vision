import numpy as np
import math
import skimage.color
import skimage.io
from skimage.transform import warp
from scipy import linalg
import random
import matplotlib.pyplot as plt
from skimage.transform import resize

# Q 3.1
def computeH(l1, l2):
    H2to1 = np.eye(3)
    # YOUR CODE HERE
    H_comp = np.array([])
    # Swap
    L_1, L_2 = l2, l1
    l1, l2 = L_1, L_2

    comp = 0
    A = l1[0,0]
    B = l1[0,1]
    C = l2[0,0]
    D = l2[0,1]
    H_comp = np.array([
        [-A,
        -B,
        -1, 
        0, 
        0, 
        0,
        A*C,
        B*C,
        C ],

        [0, 
        0, 
        0, 
        -A,
        -B,
        -1,
        A*D,
        B*D,
        D ]
        ])

    for i in range(1,l1.shape[0]):
        a = l1[i,0]
        b = l1[i,1]
        c = l2[i,0]
        d = l2[i,1]
        comp = np.array([
            [-a,
             -b,
             -1, 0, 0, 0,
             a*c,
             b*c,
             c],

            [0, 0, 0,
             -a,
             -b,
             -1,
             a*d,
             b*d,
             d ]
            ])
        H_comp = np.concatenate([H_comp, comp])
    # Identity
    svd_var = np.dot(H_comp.T, H_comp)
    _,_,V = np.linalg.svd(svd_var)
    H2to1 = np.reshape(V[8,:], [3,3])
    return H2to1

# Q 3.2
def computeHnorm(l1, l2):
    H2to1 = np.eye(3)
    # YOUR CODE HERE

    l_s = l1.shape[0]
    l_s2 = l2.shape[0]

    # Translate the mean to point of origin
    t1 = l1 - l1.mean(axis = 0)
    t2 = l2 - l2.mean(axis = 0)

    # Compute the max euclidean distance:
    euclids = []
    for i in range(l_s):
        euclid = np.sqrt( (t1[i,0])**2 + (t1[i,1])**2 )
        euclids.append(euclid)
    # Find index of largest value
    largest_l1 = euclids.index(max(euclids))
    #print (largest_l1)

    euclids = []
    for i in range(l_s2):
        euclid = np.sqrt( (t2[i,0])**2 + (t2[i,1])**2 )
        euclids.append(euclid)
    # Sort the array from smallest to biggest
    largest_l2 = euclids.index(max(euclids))
    #print (largest_l2)

    # Scale the points so that the largest ditance to the origin is sqrt(2)
    max_euclid_1 = np.sqrt( t1[largest_l1,0]**2 + t1[largest_l1,1]**2 )
    scaled_1 = max_euclid_1 / np.sqrt(2)
    max_euclid_2 = np.sqrt( t2[largest_l2,0]**2 + t2[largest_l2,1]**2 )
    scaled_2 = max_euclid_2/np.sqrt(2)

    # Now find the homogenous transformation matrix
    H_from2to1 = computeH(t1/scaled_1,t2/scaled_2)

    # Compute Homography for the other 2 coordinate frames
    H1 = computeH(l1, t1/scaled_1)
    H2 = computeH(t2/scaled_2, l2)

    # Compute the transformation based on all of these transformations
    H1_to_H21 = np.dot(H1, H_from2to1)
    H2to1 = np.dot(H1_to_H21, H2)

    return H2to1

# Q 3.3
def computeHransac(locs1, locs2):
    bestH2to1, inliers = None, None
    # YOUR CODE HERE
    l_s = locs1.shape[0]
    '''
    We initialise the parameters where:
    num = amount of points needed to fit the model
    ratio = outlier ratio (0.25 = 1 outlier out of every 4 data points)
    prob = probability that we set for there being at least one point is an inlier (how much slack we give)
    '''
    num, ratio, prob = 4, 0.2, 0.99
    m = 0

    d = 1-pow(ratio, num)
    denom = math.log(d)
    numerator = math.log(1-prob)
    
    # Round up
    #sd = math.sqrt(d)/(-1+d)

    k = int( math.ceil(numerator/denom) )
    for i in range(int(k*1.5)):

        # Randomly choose points and compute the homogenous transformation matrix
        ind = random.sample(range(l_s),  num)
        H2to1 = computeHnorm( locs1[ind],locs2[ind] )

        # Compute the projected coordinates of the points
        #print (locs1.shape)
        #print (H2to1.shape)
        #print ("!!!!!!!!!!!!!!!!!!!!!!!!")
        x_1 = np.dot(locs2, H2to1.T)
        x_shape = x_1.shape[0]
        #array_x = np.array([])
        #for i in range(x_shape):
        #    array_x[i] = x_1[i,:] / x_1[i,2]

        x_1 = np.array([ x_1[j,:] / x_1[j,2] for j in range(x_shape) ])

        currentInliners = [1 if np.sqrt( np.sum( ((x_1[k,0]-locs1[k,0])**2,(x_1[k,1]-locs1[k,1])**2))) < 10 else 0 for k in range(locs1.shape[0])]
        if sum(currentInliners) >= m:
            bestH2to1 = H2to1
            inliers = currentInliners
            m = sum(currentInliners)

        print ("Loop: " + str(i) + " out of: " + str(int(k*1.5)) )
    inliers_index = np.nonzero(inliers)
    bestH2to1 = computeHnorm(locs1[inliers_index,:][0],locs2[inliers_index,:][0])
    return bestH2to1, inliers

# Q3.4
# skimage.transform.warp will help
def compositeH( H2to1, template, img ):

    # YOUR CODE HERE
    compositeimg = warp(template, np.linalg.inv(H2to1), output_shape = img.shape)
    img = warp(img, np.eye(3), output_shape = img.shape)
    compositeimg[np.where(compositeimg == 0.0)] = img[np.where(compositeimg == 0.0)]
    return compositeimg


def HarryPotterize():
    # we use ORB descriptors but you can use something else
    from skimage.feature import ORB,match_descriptors
    # YOUR CODE HERE
    im1 = skimage.io.imread('../data/cv_cover.jpg')
    gray1 = skimage.color.rgb2gray(im1)

    im2 = skimage.io.imread('../data/cv_desk.png')
    gray2 = skimage.color.rgb2gray(im2)

    im3= skimage.io.imread('../data/hp_cover.jpg')

    # Specifify the orb describtor
    orb = ORB(n_keypoints = 3000)

    # Detect features from image 1
    orb.detect_and_extract(gray1)
    locations_1 = orb.keypoints
    descriptors_1 = orb.descriptors
    
    # Defect features from image 2
    orb.detect_and_extract(gray2)
    locations_2 = orb.keypoints
    descriptors_2 = orb.descriptors
    
    # Match descriptors
    m = match_descriptors(descriptors_1, descriptors_2)

    # Location of matching objects in first image
    l_m_1 = locations_1[ m[:,0] ]
    locs0 = np.flip(l_m_1, 1)
    one_array_1 = np.ones( [locs0.shape[0], 1] )
    locations_first = np.hstack( (locs0,one_array_1) )

    # Location of matching objects in second image
    l_m_2 = locations_2[ m[:,1] ]
    locs1 = np.flip(l_m_2,1)
    one_array_2 = np.ones( [locs1.shape[0], 1] )
    locations_second = np.hstack( (locs1,one_array_2) )

    # Compute the ransac now
    bestH2to1, inliers = computeHransac(locations_second, locations_first)

    # Generate a composite image
    composite_img = compositeH( bestH2to1, resize(im3,im1.shape),im2 )
    skimage.io.imshow(composite_img)
    skimage.io.show()

    # Print the final H Matrix
    H_final = bestH2to1 / bestH2to1[2,2]
    print(H_final)

    return
