import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.patches as patches
from SubtractDominantMotion import SubtractDominantMotion

# write your script here, we recommend the above libraries for making your animation

car = np.load('../data/aerialseq.npy')
indeces = [29, 59, 89, 119]
print (car.shape[2])
for i in range(0, car.shape[2], 1):
    print (i)
    fig,ax = plt.subplots()

    mask = SubtractDominantMotion(car[:,:,i],  car[:,:,i+1])
        
    im = np.dstack((car[:,:,i],
    	car[:,:,i],
    	car[:,:,i]))
    im_1 = np.dstack((car[:,:,i],
    	car[:,:,i],
    	car[:,:,i]))

    im[mask>0] = [0,1,0]
    if (i == 30):  print ('Frame ' + str(i))
    if (i == 60):  print ('Frame ' + str(i))
    if (i == 90):  print ('Frame ' + str(i))
    if (i == 120): print ('Frame ' + str(i))
    plt.imshow(im_1)
    plt.imshow(im, alpha=0.5)
    plt.draw()
    plt.pause(0.01)
    plt.close()