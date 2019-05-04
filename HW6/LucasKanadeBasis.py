import numpy as np
from scipy.interpolate import RectBivariateSpline as spline
from numpy.linalg import inv as inv

def LucasKanadeBasis(It, It1, rect, bases):
	# Input: 
	#	It: template image
	#	It1: Current image
	#	rect: Current position of the car
	#	(top left, bot right coordinates)
	#	bases: [n, m, k] where nxm is the size of the template.
	# Output:
	#	p: movement vector [dp_x, dp_y]

    # Put your implementation here
    
    dp = np.array([[100.0],[100.0]])
    p0 = np.array([0.0, 0.0])

    p = p0
    
    # Get the frame for each gradient and for image2
    dx, dy = np.gradient(It1,axis=1), np.gradient(It1,axis=0)
    
    Yx,Xx = dx.shape
    Yy,Xy = dy.shape
    Y , X = It1.shape 
    It = It.flatten()

    dx_mesh = spline(np.linspace(0, Yx, num=Yx, endpoint=False), 
        np.linspace(0, Xx, num=Xx, endpoint=False), dx)

    dy_mesh = spline(np.linspace(0, Yy, num=Yy, endpoint=False), 
        np.linspace(0, Xy, num=Xy, endpoint=False), dy)

    It1_mesh = spline(np.linspace(0, Y, num=Y, endpoint=False), 
        np.linspace(0, X,  num=X,  endpoint=False), It1)
    
    x1,y1,x2,y2 = rect[0],rect[1],rect[2],rect[3]
    rect_x = np.linspace(x1,x2, round(x2-x1)+1)
    rect_y = np.linspace(y1,y2, round(y2-y1)+1)
    
    converge = False
    
    while(converge == False):
    	
        Bases, gamma = [], []
        x = rect_x + p[0]
        y = rect_y + p[1]
        dx_mesh1 = dx_mesh(y,x).flatten()
        dy_mesh1 = dy_mesh(y,x).flatten()

        It1_warp = It1_mesh(y,x).flatten()
        er = It1_warp - It
        
        for i in range(bases.shape[2]):
            b = bases[:,:,i].flatten()
            Bases.append(b)
            gamma.append(np.sum(b*er))

        vstack = np.vstack((dx_mesh1, dy_mesh1))
        I_grad = np.transpose(vstack)
        gamma_B = np.sum(np.vstack(Bases)*np.vstack(gamma), 0)
        er_b = (It+gamma_B) - It1_warp
        sd = np.matmul(I_grad, np.eye(2)) 
        t_sd = np.transpose(sd)
        
        Hessian = np.matmul(t_sd,sd)
        dp = np.matmul(inv(Hessian), np.matmul(t_sd,er_b) )
        p = dp + p

        if (np.linalg.norm(dp) < 0.01):
            converge = True
    return p