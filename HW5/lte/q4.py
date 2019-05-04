import numpy as np

import skimage
import skimage.measure as mes
import skimage.color
import skimage.restoration
import skimage.filters
import skimage.morphology as morph
import skimage.segmentation

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# takes a color image
# returns a list of bounding boxes and black_and_white image
def findLetters(image):
    bboxes = []
    bw = None
    # insert processing in here
    # one idea estimate noise -> denoise -> greyscale -> threshold -> morphology -> label -> skip small boxes 
    # this can be 10 to 15 lines of code using skimage functions
    
    # 1) Grayscale image
    gray = skimage.color.rgb2gray(image)
    # 2) Blur image
    blur = skimage.filters.gaussian(gray, sigma=1)
    # 3) Threshold it
    threshold = (blur < np.mean(blur)/2.0)
    morphology = morph.dilation(
    	threshold.astype(float), 
    	morph.square(10))
    morphology = morphology.astype(int)
    #plt.imshow(morphology)
    #plt.show()


    thr = []
    area_threshold, area_avg = 150, 0
    for spot in mes.regionprops(mes.label(morphology, connectivity=2)):
        if spot.area < area_threshold:
            continue
        else:
            lr, ll, mr, ml = spot.bbox
            bx = [spot.bbox]
            r = mr-lr
            print (r)
            if(spot.area > 20): thr.append(bx)
            l = ml-ll
            my = mes.regionprops(mes.label(morphology, connectivity=1))

            area_avg = area_avg + r*l
            
            rect = mpatches.Rectangle((ll, lr), l, r, fill=False)
            bboxes.append(bx)

    area_avg = area_avg/(float(len(bboxes)))

                

    return bboxes, bw