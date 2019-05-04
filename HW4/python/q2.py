import numpy as np
from scipy.optimize import fsolve

# Q 2.1
def eightpoint(pts1, pts2, M=1):
    F = None
    
    # Start off by scaling the coordinates by M
    pts1 = pts1/M
    pts2 = pts2/M

    # Create the 9 row matrix in question
    matrix = np.zeros( (pts2.shape[0], 9) )
    # define the values that will be going in there
    list_of_values = [
    pts1[:,0]*pts2[:,0],    pts1[:,0]*pts2[:,1],    pts1[:,0], 
    pts1[:,1]*pts2[:,0],    pts1[:,1]*pts2[:,1],    pts1[:,1],
    pts2[:,0], 				pts2[:,1], 				1
    ]
    # Replace all the values with the ones that need to be added in there
    for i in range(0, len(list_of_values), 1):
    	matrix[:,i] = list_of_values[i]

    # Run an SVD operation on the dot product of the the 
    # transpose of the matrix as well as the matrix
    dot_product = np.dot( matrix.T, matrix )

    # Compute the SVD for reshaped unitary matrix that comes from the dot product
    # mentionde above. 
    # this is copied and pasted from docs.scipy
    _, _, vh = np.linalg.svd( dot_product , compute_uv = True)
    u, s, vh = np.linalg.svd( np.reshape( vh.T[:,-1], (3, 3) ) , compute_uv = True )

    # Compute the dot product of the output of this SVD computation,
    # but before you do, you want to make sure that rank of the matrix
    # to be output is 2, so you should ensure that its element is 0, to annul
    # what could lead to a non 2 matrix rank of this dot product of all these 
    # variables.
    if (s[-1] !=0): s[-1]=0
    # first multiply u and s
    prod_UxS = u * s
    dot_product_2 = np.dot(prod_UxS, vh)

    # Now you want to to the do the dot product of this dot_product with a scaling
    # matrix that is 3x3. You therefore need to reshape this dot_product_2 output as
    # a 3x3 matrix before dot producting it with the scaling matrix:
    reshaped = np.reshape(dot_product_2, (3,3) )
    scale_matrix = np.array(
    	[[float(1)/M,          0,          0],
    	[ 0,          float(1)/M,          0],
    	[ 0,                   0,   float(1)]]
    	)
    dot_product_3 = np.dot( scale_matrix.T, reshaped )
    
    # Finally compute F
    F = np.dot(dot_product_3, scale_matrix)


    return F


# Q 2.2
# you'll probably want fsolve
def sevenpoint(pts1, pts2, M=1):
    F = None
    Fs = []	# list of Fs
    N = pts1.shape[0]
    assert(pts1.shape[0] == 7)
    assert(pts2.shape[0] == 7)
    
    # ------------------------- Like before -------------------------------
    # Start off by scaling the coordinates by M
    pts1 = pts1/M
    pts2 = pts2/M

    # Create the 9 row matrix in question
    matrix = np.zeros( (pts2.shape[0], 9) )
    # define the values that will be going in there
    list_of_values = [
    pts1[:,0]*pts2[:,0],    pts1[:,0]*pts2[:,1],    pts1[:,0], 
    pts1[:,1]*pts2[:,0],    pts1[:,1]*pts2[:,1],    pts1[:,1],
    pts2[:,0], 				pts2[:,1], 				1
    ]
    # Replace all the values with the ones that need to be added in there
    for i in range(0, len(list_of_values), 1):
    	matrix[:,i] = list_of_values[i]

    # Run an SVD operation on the dot product of the the 
    # transpose of the matrix as well as the matrix
    dot_product = np.dot( matrix.T, matrix )

    # Compute the SVD for reshaped unitary matrix that comes from the dot product
    # mentionde above. 
    # this is copied and pasted from docs.scipy
    _, _, vh = np.linalg.svd( dot_product , compute_uv = True)
    print (vh)

    # ----------- Here is where it is different --------------------------
    # Here we need to find the zeros of the determinant of:
    # (k)*(last row of the transpose of vh from above) 
    #							+
    # (1-k)(second last row of the tranpose of vh from above)

    # Since we are using scipy, let us use the function solving tool fsolve
    # to find what values of a give us an output of 0
    list_of_k = [0,0,0]
    # import sympy <-- WOrks like shit
    def f(k):
	    k_vh      = k*np.reshape( vh.T[:,-1], (3,3) )
	    k_vh2 = (1-k)*np.reshape( vh.T[:,-2], (3,3) )
	    arguement = k_vh + k_vh2
	    det = np.linalg.det(arguement)

	    return det

	# -----------------------------    
    print ("Now begin finding the")
    
    for value in range(-1000, 1000, 1):
    	k, _, i, _ = fsolve(f, x0=float(value)/100, full_output=1)
    	if (k not in list_of_k): list_of_k.append(k)
    	if len(list_of_k) == 3: break
    # ------------------------------
    print (len(list_of_k))

    # Same shit all over again, except with more Fs
    scale_matrix = np.array(
    	[[float(1)/M,          0,          0],
    	[ 0,          float(1)/M,          0],
    	[ 0,                   0,   float(1)]]
    	)

    # Compute F for all of them, and add them to the list
    j = 0
    for k in list_of_k:
    	f = 0
    	k_vh      = k*np.reshape( vh.T[:,-1], (3,3) )
    	k_vh2 = (1-k)*np.reshape( vh.T[:,-2], (3,3) )
    	# Compute the dot pro
    	dot_product_4 = np.dot(scale_matrix.T, k_vh+k_vh2)
    	#print (dot_product_4)
    	# Bring it back to the shape it needs before appending it
    	F = np.dot(dot_product_4, scale_matrix)
    	Fs.append(F)
    	j += 1
    F = Fs[400]

    return F
