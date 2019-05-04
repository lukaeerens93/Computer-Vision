import numpy as np
from q2 import eightpoint, sevenpoint
from q3 import triangulate
# Q 5.1
# we're going to also return inliers
def ransacF(pts1, pts2, M):
    F = None
    inliers = None
    
    return F, inliers

# Q 5.2
# r is a unit vector scaled by a rotation around that axis
# 3x3 rotatation matrix is R
# https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
# http://www.cs.rpi.edu/~trink/Courses/RobotManipulation/lectures/lecture6.pdf
def rodrigues(r):
    R = None
    

    return R


# Q 5.2
# rotations about x,y,z is r
# 3x3 rotatation matrix is R
# https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
# https://en.wikipedia.org/wiki/Axis%E2%80%93angle_representation
def invRodrigues(R):
    r = None
    
    return r

# Q5.3
# we're using numerical gradients here
# but one of the great things about the above formulation
# is it has a nice form of analytical gradient
def rodriguesResidual(K1, M1, p1, K2, p2, x):
    residuals = None
    
    return residuals

# we should use scipy.optimize.minimize
# L-BFGS-B is good, feel free to use others and report results
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
def bundleAdjustment(K1, M1, p1, K2, M2init, p2,Pinit):
    M2, P = None, None
    
    return M2,P 