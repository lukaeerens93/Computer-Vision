import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import affine_transform as af_trans

from numpy.linalg import inv as inv 


def InverseCompositionAffine(It, It1):
	# Input: 
	#	It: template image
	#	It1: Current image

	# Output:
	#	M: the Affine warp matrix [2x3 numpy array]

    # put your implementation here
    M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    p  = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    dp = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

    Y,X = It.shape

    
    x,y = np.linspace(0,X,X), np.linspace(0,Y,Y)
    yt,xt = np.meshgrid(y.astype('int'),
        x.astype('int'),sparse=False,indexing='ij')
    xt = xt.flatten()
    yt = yt.flatten()
    print (len)

    # ---------- Was in the while loop inregular affine, now is here -----------------------
    # Compute gradient
    dx = np.gradient(It1,axis=1)
    dy = np.gradient(It1,axis=0)
    # Try to write this using Affine transform instead of shift and compute gradient of img
    w_dx, w_dy = af_trans(dx, M), af_trans(dy, M)
    It_dx, It_dy = w_dx.flatten(), w_dy.flatten()
    # Find steepest descent (largest magnitude) and compute Hessian
    sd = np.vstack((xt*It_dx, yt*It_dx, It_dx, 
                    xt*It_dy, yt*It_dy, It_dy)).T
    Hessian = np.matmul(sd.T, sd)
    #print (Hessian)


    converge = False

    #c = 0
    while(converge == False):
        #print (c)
        # Warp and compute the error
        It_mask = It*af_trans( np.ones((Y,X)), M)
        It1_w = af_trans(It1,M)
        er = It1_w - It_mask	# We now swap the error
        er = er.flatten()
        #print (er)

        vstack = np.vstack((It_dx,It_dy))
        er_dec = np.matmul(vstack, er)
        sd_u = np.matmul(sd.T, er)

        dp = np.matmul(inv(Hessian), sd_u)

        p = p+dp
        # Watch out baby, we now add dp not p!
        M_ = np.array([  [1+dp[4],    dp[3],   dp[5] ],   
                         [  dp[1],  1+dp[0],   dp[2] ], 
                         [     0,	    0,	   1  ]  ])
        M = np.dot(M, inv(M_))
        

        if (np.linalg.norm(dp) < 0.01):
        	converge = True
        #c+=1

    return M
