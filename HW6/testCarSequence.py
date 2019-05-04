import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.patches as patches
from LucasKanade import LucasKanade
import time


# write your script here, we recommend the above libraries for making your animation
car = np.load('../data/carseq.npy')
#print (car.shape)
x1,x2,y1,y2 = 59,145,116,151
rectangles = []

def update(c_x, c_y, rec, index):
	print ('Frame ' + str(index))
	plt.plot(c_x,c_y, 'r-', linewidth = 2)
	rectangles.append(rec)
	time.sleep(2)

for i in range(0, car.shape[2]-1, 1):
	coord_x, coord_y = [x1,x2,x2,x1,x1], [y1,y1,y2,y2,y1]

	frame = car[y1:y2+1, x1:x2+1, i]
	print (i)
	fig,ax = plt.subplots(1)
	ax.imshow(car[:,:,i])
	#plt.imshow(car[:,:,i])
	#plt.imshow(frame)
	#plt.draw()
	#plt.pause(0.01)
	rect = [x1,y1,x2,y2]
	r1 = patches.Rectangle((x1,y1),
		x2-x1,y2-y1,linewidth=1,edgecolor='r', facecolor='none')
	#plt.plot(coord_x,coord_y, 'r-', linewidth = 2)
	ax.add_patch(r1)
	print (rect)
	plt.draw()
	plt.pause(0.01)
	plt.close()
	

	# Compute Lucas Kanade for the current frame and then the frame right after that
	p = LucasKanade(frame, car[:,:,i+1], rect)
	#print (p)
	if (i == 1): update(coord_x,coord_y,rect,i)   
	if (i == 100): update(coord_x,coord_y,rect,i)
	if (i == 200): update(coord_x,coord_y,rect,i)
	if (i == 300): update(coord_x,coord_y,rect,i)
	if (i == 400): update(coord_x,coord_y,rect,i)

	

	x1-=round(p[0])
	x2-=round(p[0])
	y1-=round(p[1])
	y2-=round(p[1])
	#x1, x2 = int(np.ceil(x1)), int(np.ceil(x2))
	#y1, y2 = int(np.ceil(y1)), int(np.ceil(y2))

	x1, x2 = int(x1), int(x2)
	y1, y2 = int(y1), int(y2)

	coord_x = [x1,x2,x2,x1,x1]
	coord_y = [y1,y1,y2,y2,y1]
print ('Done')
np.save('../data/carseqrects.npy', rectangles)
print ('Rectangles Saved')
