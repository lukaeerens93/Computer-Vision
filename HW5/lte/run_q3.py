import numpy as np
import scipy.io
import copy
from nn import *

train_data = scipy.io.loadmat('../data/nist36_train.mat')
valid_data = scipy.io.loadmat('../data/nist36_valid.mat')
i_list, tl_list, ta_list, vl_list, va_list = [], [], [], [], []
train_x, train_y = train_data['train_data'], train_data['train_labels']
valid_x, valid_y = valid_data['valid_data'], valid_data['valid_labels']

max_iters = 50
# pick a batch size, learning rate
batch_size = 128
learning_rate = 0.01
hidden_size = 64

batches = get_random_batches(train_x,train_y,batch_size)
batch_num = len(batches)

params = {}

# initialize layers here
initialize_weights(1024,64,params,'layer1')
initialize_weights(64,36,params,'output')
first_params = copy.deepcopy(params)

# with default settings, you should get loss < 150 and accuracy > 80%
for itr in range(max_iters):
    total_loss = 0
    total_acc = 0
    for xb,yb in batches:
        #pass
        # training loop can be exactly the same as q2!
        # forward
        layer1 = forward(    xb, params, 'layer1')
        output = forward(layer1, params, 'output', softmax)

        # loss
        # be sure to add loss and accuracy to epoch totals 
        loss, acc = compute_loss_and_acc(yb, output)
        # Incremement the total loss and the total accuracy
        total_loss = total_loss + loss
        total_acc  = total_acc  + acc
     
        # backward
        dx_x, dx_y = np.nonzero(yb)
        delta1 = output
        delta1[dx_x, dx_y] -= 1
        delta2 = backwards(delta1, params, 'output', linear_deriv)
        backwards(delta2, params, 'layer1', sigmoid_deriv)

        # apply gradient
        params['Woutput'] = params['Woutput'] - params['grad_Woutput']*learning_rate
        params['boutput'] = params['boutput'] - params['grad_boutput']*learning_rate

        params['Wlayer1'] = params['Wlayer1'] - params['grad_Wlayer1']*learning_rate
        params['blayer1'] = params['blayer1'] - params['grad_blayer1']*learning_rate
    total_acc = total_acc/batch_num
    batcH = batch_num*batch_size
    total_loss = total_loss/batcH

    if itr % 2 == 0:
        print("itr: {:02d} \t loss: {:.2f} \t acc : {:.2f}".format(itr,total_loss,total_acc))
        i_list.append(itr)
        ta_list.append(total_acc)
        tl_list.append(total_loss)
        
        layer1 = forward(valid_x, params, 'layer1')
        output = forward(layer1, params, 'output', softmax)
        valid_loss, valid_acc = compute_loss_and_acc(valid_y, output)
        va_list.append(valid_acc)
        vl_list.append(valid_loss/ valid_x.shape[0])
# run on validation set and report accuracy! should be above 75%
print (tl_list)
valid_acc = None
layer1 = forward(valid_x, params, 'layer1')
output = forward(layer1, params, 'output', softmax)
valid_loss, valid_acc = compute_loss_and_acc(valid_y, output)
print('Validation accuracy: ',valid_acc)

if False: # view the data
    for crop in xb:
        import matplotlib.pyplot as plt
        plt.imshow(crop.reshape(32,32).T)
        plt.show()
import pickle
saved_params = {k:v for k,v in params.items() if '_' not in k}
with open('q3_weights.pickle', 'wb') as handle:
    pickle.dump(saved_params, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Q3.1.3
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
#---------------------
plt.figure(1)
plt.plot(i_list, tl_list, 'b', label='Training')
plt.plot(i_list, vl_list, 'r', label='Validation')
plt.title('Losses')
plt.legend()
plt.show()

plt.figure(1)
plt.plot(i_list, ta_list, 'b', label='Training')
plt.plot(i_list, va_list, 'r', label='Validation')
plt.title('Accuracy')
plt.legend()
plt.show()

#---------------------

F = plt.figure(1, (22, 10))
f_p, p = first_params['Wlayer1'], params['Wlayer1']
vmax = f_p.reshape((32,32,-1)).max()
vmin = f_p.reshape((32,32,-1)).min()
grid = ImageGrid(F, 121, nrows_ncols=(8,8), axes_pad=0.1, add_all=True, label_mode="L")
for i in range(hidden_size): grid[i].imshow( f_p.reshape((32,32,-1))[:,:,i], vmin=vmin, vmax=vmax, interpolation='nearest')
plt.draw()
vmax = p.reshape((32,32,-1)).max()
vmin = p.reshape((32,32,-1)).min()
grid = ImageGrid(F, 122, nrows_ncols=(8,8), axes_pad=0.1, add_all=True, label_mode="L")
for i in range(hidden_size): grid[i].imshow(p.reshape((32,32,-1))[:,:,i], vmin=vmin, vmax=vmax, interpolation='nearest')
plt.draw()
plt.show()



# Q3.1.3
confusion_matrix = np.zeros((train_y.shape[1],train_y.shape[1]))
for i in range(np.argmax(valid_y, axis=1).shape[0]):
    confusion_matrix[ np.argmax(valid_y, axis=1)[i], np.argmax(output, axis=1)[i] ] += 1

import string
plt.imshow(confusion_matrix,interpolation='nearest')
plt.grid(True)
plt.xticks(np.arange(36),string.ascii_uppercase[:26] + ''.join([str(_) for _ in range(10)]))
plt.yticks(np.arange(36),string.ascii_uppercase[:26] + ''.join([str(_) for _ in range(10)]))
plt.show()