import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import affine_transform as af_trans
from numpy.linalg import inv as inv 

def LucasKanadeAffine(It, It1):
    # Input: 
    #    It: template image
    #    It1: Current image
    # Output:
    #    M: the Affine warp matrix [2x3 numpy array]
    # put your implementation here
    M  = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    p  = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    dp = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

    Y,X = It.shape

    
    x,y = np.linspace(0,X,X), np.linspace(0,Y,Y)
    yt,xt = np.meshgrid(y.astype('int'),
        x.astype('int'),sparse=False,indexing='ij')
    xt = xt.flatten()
    yt = yt.flatten()
    print (len)

    converge = False

    c = 0
    while(converge == False):
        print (c)
        # Warp and compute the error
        It_mask = It*af_trans( np.ones((Y,X)), M)
        It1_w = af_trans(It1,M)
        er = It_mask - It1_w
        er = er.flatten()

        # Compute gradient
        dx = np.gradient(It1,axis=1)
        dy = np.gradient(It1,axis=0)

        # Try to write this using Affine transform instead of shift and compute gradient of img
        w_dx, w_dy = af_trans(dx, M), af_trans(dy, M)
        It_dx, It_dy = w_dx.flatten(), w_dy.flatten()
        vstack = np.vstack((It_dx,It_dy))
        er_dec = np.matmul(vstack, er)
        
        # Find steepest descent (largest magnitude) and compute Hessian
        sd = np.vstack((xt*It_dx, yt*It_dx, It_dx, 
                        xt*It_dy, yt*It_dy, It_dy)).T
        sd_u = np.matmul(sd.T, er)
        Hessian = np.matmul(sd.T, sd)

        dp = np.matmul(inv(Hessian), sd_u)

        p = p+dp
        M = np.array([  [1+p[4],  p[3], p[5] ],   
                        [p[1],  1+p[0], p[2] ]  ])

        if (np.linalg.norm(dp) < 0.01):
        	converge = True
        c+=1
        
    return M