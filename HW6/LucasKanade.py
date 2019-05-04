import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy import ndimage
from scipy.ndimage import shift
from numpy.linalg import inv as inv


def LucasKanade(It, It1, rect, p0 = np.zeros(2)):
	# Input: 
	#	It: template image
	#	It1: Current image
	#	rect: Current position of the car
	#	(top left, bot right coordinates)
	#	p0: Initial movement vector [dp_x0, dp_y0]
	# Output:
	#	p: movement vector [dp_x, dp_y]
	
    # Put your implementation here
    dp = np.array([100.0, 100.0])

    p = p0
    converge = False

    while (converge == False):

        # Warp the image and compute error
        x1,y1,x2,y2 = rect[0],rect[1],rect[2],rect[3]
        p_x, p_y = p[0], p[1]
        w = shift(It1,(p_y, p_x),It1.dtype)
        It_w = w[y1:y2+1, x1:x2+1]
        er = It_w - It
        er = er.flatten()

        # 3) Compute the gradient and warp them, then demarcate the border
        dx = ndimage.sobel(It1, 1)  # horizontal derivative
        dy = ndimage.sobel(It1, 0)  # vertical derivative
        w_dx = shift(dx,(p_y, p_x), It1.dtype)
        It_dx = w_dx[y1:y2+1, x1:x2+1]
        It_dx = It_dx.flatten()
        w_dy = shift(dy,(p_y, p_x), It1.dtype)
        It_dy = w_dy[y1:y2+1, x1:x2+1]
        It_dy = It_dy.flatten()
        vstack = np.vstack((It_dx, It_dy))
        er_dec = np.matmul(vstack, er)# Error of steepest descent
        
        # Compute hessian and find delta p (dp)
        l_mag = np.matmul(vstack.T, np.eye(2))
        Hessian = np.matmul(l_mag.T, l_mag)
        dp = np.matmul(inv(Hessian), er_dec)
        p = dp + p

        if (np.linalg.norm(dp) < 0.01):
        	converge = True
    return p
