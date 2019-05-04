import numpy as np
import scipy.io
from nn import *
from collections import Counter
i_list, tl_list, ta_list, vl_list, va_list = [], [], [], [], []
train_data = scipy.io.loadmat('../data/nist36_train.mat')
valid_data = scipy.io.loadmat('../data/nist36_valid.mat')

# we don't need labels now!
train_x = train_data['train_data']
valid_x = valid_data['valid_data']

max_iters = 100
# pick a batch size, learning rate
batch_size = 36 
learning_rate =  0.000005
hidden_size = 32
lr_rate = 20
batches = get_random_batches(train_x,np.ones((train_x.shape[0],1)),batch_size)
batch_num = len(batches)
adjust = batch_num * batch_size
params = Counter()
# initialize layers here

def initialize_momentum(in_size,out_size,params,name=''):
    W, b = None, None
    W = np.random.uniform(
        low  = -np.sqrt( 6.0 / (in_size + out_size) ), 
        high =  np.sqrt( 6.0 / (in_size + out_size) ), 
        size =  (in_size, out_size)
        )
    b = np.zeros((out_size))
    params['W' + name] = W
    params['b' + name] = b
    params['m_W' + name] = np.zeros( (in_size, out_size) )
    params['m_b' + name] = b

initialize_momentum(1024,32,params,'layer1')
initialize_momentum(32,32,  params,'layer2')
initialize_momentum(32,32,  params,'layer3')
initialize_momentum(32,1024,params,'output')

# with default settings, you should get loss < 150 and accuracy > 80%
for itr in range(max_iters):
    total_loss = 0
    for xb,_ in batches:
        pass
        # training loop can be exactly the same as q2!
        # your loss is now squared error
        # delta is the d/dx of (x-y)^2
        # to implement momentum
        #   just use 'm_'+name variables
        #   to keep a saved value over timestamps
        #   params is a Counter(), which returns a 0 if an element is missing
        #   so you should be able to write your loop without any special conditions
        layer1 = forward(    xb, params, 'layer1', relu)
        layer2 = forward(layer1, params, 'layer2', relu)
        layer3 = forward(layer2, params, 'layer3', relu)
        output = forward(layer3, params, 'output', sigmoid)

        # loss
        loss = np.sum((xb-output) * (xb-output))
        #print (xb-output)
        # be sure to add loss and accuracy to epoch totals 
        # Incremement the total loss and the total accuracy
        total_loss = total_loss + loss
     
        for k, v in params.items():
            if '_' in k: continue
            import math
            import scipy
            g = params['grad_' + k]*learning_rate
            params['m_'+k] = 0.9*params['m_'+k] - g
            params[k] = params[k]+ params['m_'+k]

        # backward
        deriv_loss = -1*2*(xb-output)
        deriv_3 = backwards(deriv_loss, params, 'output', sigmoid_deriv)
        deriv_2 = backwards(deriv_3, params, 'layer3', relu_deriv)
        deriv_1 = backwards(deriv_2, params, 'layer2', relu_deriv)
        backwards(deriv_1, params, 'layer1', relu_deriv)

    if itr % 2 == 0:
        print("itr: {:02d} \t loss: {:.2f}".format(itr,total_loss/adjust))
        i_list.append(itr)
        tl_list.append(total_loss/adjust)
        ta_list.append(total_loss/adjust)
        vl_list.append(total_loss)
        va_list.append(xb)
    '''
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
    '''
    if itr % lr_rate == lr_rate-1:
        learning_rate *= 0.9
import matplotlib.pyplot as plt
plt.figure(1)
plt.title('Loss')
plt.plot(i_list, tl_list)
plt.show()
# visualize some results
# Q5.3.1
xb = va_list[-1]
print (xb)
xb = valid_data['valid_data']
loops = 25
#plt.figure(1)
import matplotlib.pyplot as plt
h1 = forward(xb,params,'layer1',relu)
h2 = forward(h1,params,'hidden',relu)
h3 = forward(h2,params,'hidden2',relu)
out = forward(h3,params,'output',sigmoid)
print (out)

for i in range(5):
    choice = 500
    val = xb[i*choice]
    out = out[i*choice]
    plt.subplot(2,2,1)
    plt.imshow(val.reshape(32,32).T)
    plt.subplot(2,2,2)
    plt.imshow(val.reshape(32,32).T)
    plt.subplot(2,2,3)
    plt.imshow(out.reshape(32,32).T)
    plt.subplot(2,2,4)
    plt.imshow(out.reshape(32,32).T)
    plt.show()


from skimage.measure import compare_psnr as psnr
# evaluate PSNR
# Q5.3.2
square_error = (valid_x-out)*(valid_x-out)
smpl = 1024
MSE = np.sum(square_error,axis=1)/smpl
it2  = 10*np.log10(MSE)
PSNR = 20*np.log10(np.max(out, axis=1)) - it2
avg = np.mean(PSNR)
print('PSNR: {}'.format(avg))
