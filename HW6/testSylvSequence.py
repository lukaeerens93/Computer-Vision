import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.patches as patches

from scipy.interpolate import RectBivariateSpline as spline
from LucasKanade import LucasKanade
from LucasKanadeBasis import LucasKanadeBasis


# write your script here, we recommend the above libraries for making your animation

car = np.load('../data/sylvseq.npy')
bases = np.load('../data/sylvbases.npy')

rect = [101, 61, 155, 107]

rectangles = []
rectangles.append(rect)

p0 = np.zeros(2)

p = p0
p_l = p0


_rec = [101, 61, 155, 107]
plt.ion()

for i in range(0, car.shape[2]-1, 1):
    print (i)
    
    x1,y1,x2,y2 = rect[0],rect[1],rect[2],rect[3]
    x1_, y1_, x2_, y2_ = _rec[0],_rec[1],_rec[2],_rec[3]

    fig,ax = plt.subplots()
    coordX1 = np.linspace(x1_,x2_,round(x2_-x1_)+1)
    coordY1 = np.linspace(y1_,y2_,round(y2_-y1_)+1)
    coordX=np.linspace(x1,x2,round(x2-x1)+1)
    coordY=np.linspace(y1,y2,round(y2-y1)+1)


    Y1 , X1 = car[:,:,i].shape 
    grad_mesh = spline(np.linspace(0, Y1, num=Y1, endpoint=False), 
        np.linspace(0, X1, num=X1, endpoint=False), car[:,:,i])

    pk = LucasKanade(grad_mesh(coordY1,coordX1), car[:,:,i+1], _rec)
    pkb = LucasKanadeBasis(grad_mesh(coordY,coordX),  car[:,:,i+1], rect, bases)

    pkb[0], pk[0], pkb[1], pk[1] = round(pkb[0]), round(pk[0]), round(pkb[1]), round(pk[1])
    
    _rec = [int(x1_+pk[0]),int(y1_+pk[1]),int(x2_+pk[0]),int(y2_+pk[1])]
    
    rect = [int(x1+pkb[0]),int(y1+pkb[1]),int(x2+pkb[0]),int(y2+pkb[1])]
    rectangles.append(rect)
    x1,y1,x2,y2 = rect[0],rect[1],rect[2],rect[3]
    x1_, y1_, x2_, y2_ = _rec[0],_rec[1],_rec[2],_rec[3]

    if (i == 1):   print ("Frame " + str(i))
    if (i == 200): print ("Frame " + str(i))
    if (i == 300): print ("Frame " + str(i))
    if (i == 350): print ("Frame " + str(i))
    if (i == 400): print ("Frame " + str(i))

    ax.imshow(car[:,:,i+1])
    rect_LKB = patches.Rectangle( (x1,y1), x2-x1+1, y2-y1+1, linewidth=2,edgecolor='b',facecolor='none')
    rect_LK = patches.Rectangle((x1_,y1_), x2_-x1_+1, y2_-y1_+1, linewidth=2,edgecolor='r',facecolor='none')
        
    ax.add_patch(rect_LK)
    ax.add_patch(rect_LKB)

    
    plt.show()
    plt.pause(0.01)
    plt.close()

rectangles = np.vstack(rectangles)
np.save('../data/sylvseqrects',rectangles)