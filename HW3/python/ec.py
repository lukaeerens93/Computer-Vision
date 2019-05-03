import numpy as np
# USE A SPARSE MATRIX
import scipy.sparse
from scipy.sparse.linalg import spsolve


def poissonStitch(fg,bg,mask):
    res = np.copy(bg)
    yb, xb = np.where(mask[:,:,0] == 255)
    region_size = [yb.max()-yb.min(), xb.max()-xb.min()]
    elem_num = np.prod(region_size)
    A = scipy.sparse.identity(np.prod(region_size), format='lil')
    # YOUR CODE HERE
    
    return res