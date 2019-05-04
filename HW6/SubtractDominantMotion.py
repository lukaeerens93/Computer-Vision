import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.morphology import binary_erosion
from scipy.ndimage.morphology import binary_dilation

from LucasKanadeAffine import LucasKanadeAffine
from InverseCompositionAffine import InverseCompositionAffine
from scipy.ndimage import affine_transform as af_trans

def SubtractDominantMotion(image1, image2):
	# Input:
	#	Images at time t and t+1 
	# Output:
	#	mask: [nxm]
    # put your implementation here
    
    y,x = image1.shape
    #M = LucasKanadeAffine(image1, image2)

    #--#--#--#--#--#--#--#--#--#--#--#--#--#--#--#--#--#--#--#
    # ----------- WARNING: !!!!!!!!!!!!!!! ----------------- #
    # ------------------------------------------------------ #
    # If you want to run inverse composition affine, you need
    # to comment out LucasKanadeAffine above, and uncomment 
    # the next 4 lines of code
    M = InverseCompositionAffine(image1, image2)

    #--#--#--#--#--#--#--#--#--#--#--#--#--#--#--#

    img1_mask = image1*af_trans(np.ones((y,x)), M)
    img2_aff = af_trans(image2,M)
    
    
    mask = abs(img1_mask - img2_aff)

    #print (mask)
    # What value should I make the threshold?
    # I have tried:
    #    0.1   |   0.2	 |	  0.3  
    # =============================
    #    Good  |   Ok	 |	  Shit

    mask[mask >= 0.1] = 1
    mask[mask < 0.1] = 0
    #plt.imshow(mask)
    #plt.show()
    #plt.close()
    mask = binary_erosion(mask, iterations  = 1)
    mask = binary_dilation(mask, iterations = 5)
    return mask
