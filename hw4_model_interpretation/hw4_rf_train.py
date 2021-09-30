import numpy as np
import sys

x_train = np.array([])
y_train = np.array([])
f = open(sys.argv[1],'r')
s = f.readline()
while True:
    s0 = f.readline()
    if len(s0) ==0:
        break
    ls0 = s0.split(sep=',')
    ls = ls0[1].split(sep=' ')    
    nps = np.asarray(ls,dtype=np.int)    
    if x_train.size == 0:
        x_train = np.reshape(nps,(1,48,48))
    else:
        x_train = np.vstack((x_train,np.reshape(nps,(1,48,48))))    
    npy = int(ls0[0])
    npy = np.reshape(npy,(1,1))
    if y_train.size == 0:
        y_train = npy
    else:
        y_train = np.vstack((y_train,npy))
    if x_train.shape[0]%3100==0:
        print(x_train.shape[0])
        break
x_train = np.reshape(x_train,(x_train.shape[0],48,48,1))/255
y_temp = np.zeros((y_train.shape[0],7))
for i in range(y_train.shape[0]):
    y_temp[i][y_train[i]] = 1
np.save('x_train.npy',x_train)
np.save('y_train.npy',y_temp)

x_train = None
y_train = None
y_temp = None

    