import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D

from q2 import eightpoint
from q3 import essentialMatrix, triangulate
from util import camera2
from scipy.ndimage import filters
from numpy import linalg as lin
import scipy
# Q 4.1
# np.pad may be helpful
def epipolarCorrespondence(im1, im2, F, x1, y1):
    x2, y2 = 0, 0

    # create list of coordinates
    coord = np.array([x1, y1, 1])
    epipolar_line = np.dot(F, coord)

    # Define the patch size (9 pixels, with the 5th pixel being the middle point)
    y_min, y_max = max(0, x1-5), min(x1+6, im1.shape[1]-1)
    y_min, y_max = int(y_min), int(y_max)

    x_min, x_max = max(0, y1-5), min(y1+6, im1.shape[0]-1)
    x_min, x_max = int(x_min), int(x_max)

    im_patch = im1[x_min:x_max, y_min:y_max, :]

    update = np.array([x1,y1])
    # search field is 20 pixels to each side
    A, thresh = 0, 0
    for i in range( max(5, x1-20), min(im2.shape[1]-6, x1+20) ):
        j = epipolar_line[0]*i + epipolar_line[2]
        j = int(-j / epipolar_line[1])
        if (j <= im2.shape[0]-6):
            if (j >= 5):
                dif = np.array( [i,j] ) - update
                e1 = lin.norm(dif)
                xMin, xMax, yMin, yMax = i-5, i+6, j-5, j+6
                e2 = lin.norm( im_patch - im2[yMin:yMax, xMin:xMax, :] )
                e2 /= (7**2)
                if (A == 0): thresh = 5000     # Shows how bad is doing
                if (thresh > e1+e2): x2, y2, thresh = i, j, e1+e2
        A += 1

        
    return x2, y2

# Q 4.2
# this is the "all in one" function that combines everything
def visualize(IM1_PATH,IM2_PATH,TEMPLE_CORRS,F,K1,K2):
    fig = plt.figure()
    ax = Axes3D(fig)
    # ax.set_xlim/set_ylim/set_zlim/
    # ax.set_aspect('equal')
    # may be useful
    # you'll want a roughly cubic meter
    # around the center of mass
    # Define pts and ims
    from scipy import io as Mat
    corrs = Mat.loadmat(TEMPLE_CORRS)
    pts = np.concatenate( [ corrs['x1'], corrs['y1'] ], axis=1 )
    from skimage import io as Im
    im1, im2 = Im.imread(IM1_PATH), Im.imread(IM2_PATH)
    
    # Use the epipolarCorrespondence function from before to compute the new points
    #new_pts = np.zeros([len(pts)], dtype=np.ndarray)
    new__pts = np.zeros([len(pts)], dtype=np.ndarray)
    new_pts = []
    for i in range(0, len(pts), 1):
        x1,y1 = int(corrs['x1'][i]), int(corrs['y1'][i])
        print (epipolarCorrespondence(im1, im2, F, x1, y1))
        x,y = epipolarCorrespondence(im1, im2, F, x1, y1)
        new_pts.append([ x,y ])
        new__pts[i] = [ epipolarCorrespondence(im1, im2, F, x1, y1) ]

    # From run_q3, triangulate the essential matrix (verbatim code from run_q3.py)
    E = essentialMatrix(F,K1,K2)
    E = E/E[2,2]
    M2s = camera2(E)
    #print (new_pts)

    # Q3.2 / 3.3
    C1 = np.hstack([np.eye(3),np.zeros((3,1))])
    new_pts = np.array(new_pts)
    for C2 in M2s:
        P, err = triangulate(K1.dot(C1),pts,K2.dot(C2),new_pts)
        if(P.min(0)[2] > 0): 
            break
    kc1, kc2 = np.dot(K1,C1), np.dot(K2,C2)
    scipy.io.savemat('q4_2.mat', {'F':F,'M1':C1,'M2':C2,'C1':kc1,'C2':kc2} )
    ax.scatter(P[:,0], P[:,1], P[:,2])
    plt.show()
    print( 'M2: ' + str(C2) )
    print( 'C2: ' + str(np.dot(K2,C2)) )
    

# Extra credit
def visualizeDense(IM1_PATH,IM2_PATH,TEMPLE_CORRS,F,K1,K2):

    fig = plt.figure()
    ax = Axes3D(fig)
    # ax.set_xlim/set_ylim/set_zlim/
    # ax.set_aspect('equal')
    # may be useful
    # you'll want a roughly cubic meter
    # around the center of mass
    # Define pts and ims

    '''
    SAME SHIT EXCEPT at this location (C)
    '''
    from scipy import io as Mat
    corrs = Mat.loadmat(TEMPLE_CORRS)
    pts = np.concatenate( [ corrs['x1'], corrs['y1'] ], axis=1 )
    from skimage import io as Im
    im1, im2 = Im.imread(IM1_PATH), Im.imread(IM2_PATH)
    
    # Use the epipolarCorrespondence function from before to compute the new points
    #new_pts = np.zeros([len(pts)], dtype=np.ndarray)
    new__pts = np.zeros([len(pts)], dtype=np.ndarray)
    new_pts = []
    for i in range(0, len(pts), 1):
        x1,y1 = int(corrs['x1'][i]), int(corrs['y1'][i])
        print (epipolarCorrespondence(im1, im2, F, x1, y1))
        x,y = epipolarCorrespondence(im1, im2, F, x1, y1)
        new_pts.append([ x,y ])
        new__pts[i] = [ epipolarCorrespondence(im1, im2, F, x1, y1) ]

    # From run_q3, triangulate the essential matrix (verbatim code from run_q3.py)
    E = essentialMatrix(F,K1,K2)
    E = E/E[2,2]
    M2s = camera2(E)
    #print (new_pts)

    # Q3.2 / 3.3
    C1 = np.hstack([np.eye(3),np.zeros((3,1))])
    new_pts = np.array(new_pts)
    for C2 in M2s:
        P, err = triangulate(K1.dot(C1),pts,K2.dot(C2),new_pts)
        if(P.min(0)[2] > 0): 
            break
    kc1, kc2 = np.dot(K1,C1), np.dot(K2,C2)
    scipy.io.savemat('q4_2.mat', {'F':F,'M1':C1,'M2':C2,'C1':kc1,'C2':kc2} )
    ax.scatter(P[:,0], P[:,1], P[:,2])
    plt.show()
    print( 'M2: ' + str(C2) )
    print( 'C2: ' + str(np.dot(K2,C2)) )

    return