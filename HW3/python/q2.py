import numpy as np
import scipy.io as sio
import skimage.feature
import matplotlib.pyplot as plt
import skimage.color
import skimage.io
import math
import skimage.feature as feat
import matplotlib.pyplot as plt


# Q2.1
# create a 2 x nbits sampling of integers from to to patchWidth^2
# read BRIEF paper for different sampling methods
def makeTestPattern(patchWidth, nbits):
    res = None
    # YOUR CODE HERE
    '''
    Create a vector that is (nbits x 1)
    filled with elements between 0 and pathWitdh - 1
    '''
    p_w = patchWidth**2
    compare_x = np.random.randint(-p_w/2.0, p_w/2.0, size = (1, nbits))
    compare_y = np.random.randint(-p_w/2.0, p_w/2.0, size = (1, nbits))

    compare_x = compare_x[0]
    compare_y = compare_y[0]
    # Unravelled in memory
    unrav_x = np.unravel_index(compare_x + 40, (patchWidth, patchWidth))
    unrav_y = np.unravel_index(compare_y + 40, (patchWidth, patchWidth))
    
    return compare_x, compare_y, unrav_x, unrav_y 


# Here is a function that will be a keypoint detector:
def key_point_detector(l, comp_x, image):
    # l : location of key point
    method = feat.corner_harris(image,sigma = 1.5)
    l = feat.corner_peaks(method, min_distance = 5)
    for i in range(l.shape[0]):
        # The code below was found on this link:
        # http://scikit-image.org/docs/0.13.x/api/skimage.draw.html#skimage.draw.circle_perimeter
        rr,cc = skimage.draw.circle_perimeter(l[i,0], l[i,1], 2)
        image[rr,cc] = 1
    skimage.io.imshow(image)
    skimage.io.show()
    return l


# Q2.2
# im is 1 channel image, locs are locations
# compareX and compareY are idx in patchWidth^2
# should return the new set of locs and their descs
def computeBrief(im,locs,compareX,compareY, unravelled_x, unravelled_y):
    desc = None
    # YOUR CODE HERE

    # Shape of the dimensions
    # Arrays that containt the descriptor and locations of key points
    desc, ls = np.array([]), np.array([])
    #print (im.shape)
    #print (locs.shape[0])
    #print (unravelled_x)
    #print ("!!!!!!!!!!!!!")
    for i in range(locs.shape[0]):
        x,y = im.shape[0], im.shape[1]
        a, b = locs[i,0], locs[i,1]
        u_x, u_y = unravelled_x, unravelled_y

        if (a - 4 >= 0 and a + 4 < x ):
            if (b - 4 >= 0 and b + 4 < y ):
                u_x1, u_x2 = unravelled_x[0], unravelled_x[1]
                u_y1, u_y2 = unravelled_y[0], unravelled_y[1]
                feat = [1 if im[a+u_x1[j]-4, b+u_x2[j]-4] > im[a+u_y1[j]-4, b+u_y2[j]-4] else 0 for j in range(compareX.shape[0])]
                #for j in range(compareX.shape[0]):
                #    feat = []
                #    u_x1, u_x2 = unravelled_x[0][j], unravelled_x[1][j]
                #    u_y1, u_y2 = unravelled_y[0][j], unravelled_y[1][j]
                #    im_1 = [ a+u_y1-4, b+u_y2-4 ]
                #    im_2 = [ a+u_x1-4, b+u_x2-4 ]
                #    print (im[im_1])
                #    print ("(((((((((((((((((((((((((((((((((((((")
                #    if (im[im_1] < im[im_2]):
                #        feat.append(1)
                #    else:
                #        feat.append(0)
                # Convert into a numpy array the list of features that you have just found
                feat = np.array(feat)
                ls = np.vstack( (ls, locs[i,:]) ) if ls.shape[0] != 0 else  locs[i,:]
                desc = np.vstack((desc,feat)) if desc.shape[0] !=0 else feat
  
    return ls, desc

    

# Q2.3
# im is a 1 channel image
# locs are locations
# descs are descriptors
# if using Harris corners, use a sigma of 1.5
def briefLite(im, compare_x,compare_y, unravelled_x, unravelled_y):
    locs, desc = None, None
    # YOUR CODE HERE
    method = feat.corner_harris(im, sigma = 1.5)
    locs = feat.corner_peaks(method, min_distance = 1)
    locs, desc = computeBrief(im,locs,compare_x,compare_y, unravelled_x, unravelled_y)
    return locs, desc

# Q 2.4
def briefMatch(desc1,desc2,ratio=0.8):
    # okay so we say we SAY we use the ratio test
    # which SIFT does
    # but come on, I (your humble TA), don't want to.
    # ensuring bijection is almost as good
    # maybe better
    # trust me
    matches = skimage.feature.match_descriptors(desc1,desc2,'hamming',cross_check=True)
    return matches

def plotMatches(im1,im2,matches,locs1,locs2):
    fig, ax = plt.subplots(nrows=1, ncols=1)
    skimage.feature.plot_matches(ax,im1,im2,locs1,locs2,matches,matches_color='r')
    plt.show()
    return

def testMatch(compareX, compareY, uX, uY):
    # YOUR CODE HERE

    # Load image 1 and convert to grayscale
    im1 = skimage.io.imread('../data/chickenbroth_01.jpg')
    gray1= skimage.color.rgb2gray(im1)

    # Load image 2 and convert to grayscale
    im2 = skimage.io.imread('../data/model_chickenbroth.jpg')
    gray2 = skimage.color.rgb2gray(im2)

    # Compute the Brief Life on the gray scaled images
    locs1, desc1 = briefLite(gray1, compareX, compareY, uX, uY)
    locs2, desc2 = briefLite(gray2, compareX, compareY, uX, uY)

    matches = briefMatch(desc1, desc2, ratio = 0.8)
    plotMatches(im1, im2, matches, locs1, locs2)
    return


# Q 2.5
# we're also going to use this to test our
# extra-credit rotational code
def briefRotTest(compareX, compareY, uX, uY, briefFunc):
    # you'll want this
    import skimage.transform
    # Use the same chickenbroth image

    # Load image 1 and convert to grayscale
    im1 = skimage.io.imread('../data/chickenbroth_01.jpg')
    gray1= skimage.color.rgb2gray(im1)

    # Load image 2 and convert to grayscale
    im2 = skimage.io.imread('../data/model_chickenbroth.jpg')
    gray2 = skimage.color.rgb2gray(im2)

    # Here is where the rotation angle is specified:
    rotation_angle = 10
    round_up_angle = math.ceil( 360/float(rotation_angle) ) + 1
    locs1, desc1 = briefFunc(gray1, compareX, compareY, uX, uY)
    
    # This array over here, counts the number of matches that we have
    Count = []
    c = 1
    for i in range( int(round_up_angle) ):
        print ("Epoch: " + str(c))
        gray2 = skimage.transform.rotate(gray2, rotation_angle*i)

        # Find the number of matches
        locs2,desc2 = briefFunc(gray2, compareX, compareY, uX, uY)
        matches = briefMatch(desc1, desc2, ratio=0.8)

        # Count the number of matches so that you can show this bar chart
        Count.append(len(matches))
        c += 1
    print (Count)
    Count = np.array(Count)
    plt.bar(np.arange(37), Count)
    plt.show()
    #plot_bar(Count)    
    #print(Count)
    
    #print(len(Count))

    return

# Q2.6
# YOUR CODE HERE


# put your rotationally invariant briefLite() function here
def briefRotLite(im, compareX, compareY, uX, uY,):
    locs, desc = None, None
    # YOUR CODE HERE
    method = feat.corner_harris(im,sigma = 1.5)
    locs = feat.corner_peaks(method, min_distance = 2)
    patch_width = 9

    # Load the matrices that we saved:
    comp_x = sio.loadmat('testPattern.mat')['compareX'][0]
    comp_y = sio.loadmat('testPattern.mat')['compareY'][0]

    unrav_x = np.unravel_index(comp_x + 40, (patch_width, patch_width))
    unrav_y = np.unravel_index(comp_y + 40, (patch_width, patch_width))

    # Find the identity matrix
    I = np.dot(locs.T, locs)

    # Compute the principal direction (d) after computing SVD on I
    _,_,SVD = np.linalg.svd(I)
    d = np.array( SVD[0,:] )

    # Compute rotation matrix now that you have principal direction
    R = np.array([ [d[0],d[1] ], [-d[1],d[0]] ])

    # Now that you have the rotation matrix, unravel it it to find the new location of Y
    y = np.unravel_index(comp_y + 40, (patch_width, patch_width))
    
    # Compute the dot product of the rotaiton matrix with coordinates
    y = np.dot(R, y)
    y_range = y.shape[1]

    # Update the new locations
    y_ = np.array([ 9*y[0,i] + y[1,i] for i in range(y_range) ]) - 40

    # Now that that is done, recompute the brief to try and find the keypoints
    locs, desc = computeBrief(im, locs, comp_x, comp_y, unrav_x, unrav_y)
    


    return locs, desc


