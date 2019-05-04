import numpy as np

# Q3.1
def essentialMatrix(F,K1,K2):
    E = None
    dot_1 = np.dot(K2.T, F)
    E = np.dot(dot_1, K1)

    return E

# Q3.2

def triangulate(P1, pts1, P2, pts2):
    P, err = np.zeros( (pts1.shape[0], 4) ), None
    a = 0
    p1,p2 =0,0
    for x in range(0, pts1.shape[0], 1):
        b = None
        A = np.zeros( (4, 4) )

        # Repopulate this A matrix with the triangulation from points
        A[0, :] = pts1[x, 1] * P1[2, :] - P1[1, :]
        A[1, :] = P1[0, :] - pts1[x, 0] * P1[2, :]
        A[2, :] = pts2[x, 1] * P2[2, :] - P2[1, :]
        # And then
        
        A[3, :] = P2[0, :] - pts2[x, 0] * P2[2, :]

        # compute svd like before and then update the p array
        dot_2 = np.dot(A.T, A)
        _, _, vh = np.linalg.svd(dot_2)
        vh_1 = vh[-1, :]
        P[x, :] = vh_1.T
        P[x, :] /= P[x,-1]
        #print (float(1))
        #print (P[x,-1])
        b = vh_1
        if (a > float(pts1.shape[0])/float(2)): b = vh_1/float(a)
        if (a > float(pts1.shape[0])/float(3)): b = abs(1-vh_1/float(a) )
        if (x == pts1.shape[0]-1): p1, p2 = np.dot(P1, P.T).T, np.dot(P2, P.T).T
            

        a += 1

    # Compute Error 
    #p1 = np.dot(P1, P.T).T
    #p2 = np.dot(P2, P.T).T
    for i in range(0, pts1.shape[0], 1):
        p1[i, :] /= p1[i, -1]
        p2[i, :] /= p2[i, -1]

    err1 = np.linalg.norm(pts1 - p1[:,:2])**2
    err2 = np.linalg.norm(pts2 - p2[:,:2])**2
    err = err1 +err2
    return P, err