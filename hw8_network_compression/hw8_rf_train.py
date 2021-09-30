import numpy as np
import sys

x_train = np.zeros((28709,1,48,48))
y_train = np.zeros((28709))
f = open(sys.argv[1],'r')
s = f.readline()

for i in range(28709):
    s0 = f.readline()
    if len(s0) ==0:
        break
    ls0 = s0.split(sep=',')
    ls = ls0[1].split(sep=' ')    
    nps = np.asarray(ls,dtype=np.int)
    nps = np.reshape(nps,(1,48,48))    
    x_train[i] = nps
    npy = int(ls0[0])     
    y_train[i] = npy

y_train = y_train.astype(int)
y_temp = np.zeros((28709,7))
for i in range(y_train.shape[0]):
    y_temp[i][y_train[i]] = 1
np.save('data/x_train.npy',x_train)
np.save('data/y_train.npy',y_train)

x_train = None
y_train = None
y_temp = None
"""
x_test = np.zeros((7178,1,48,48))
f = open(sys.argv[1],'r')
s = f.readline()

for i in range(7178):
    s0 = f.readline()
    if len(s0) ==0:
        break
    ls0 = s0.split(sep=',')
    ls = ls0[1].split(sep=' ')    
    nps = np.asarray(ls,dtype=np.int)
    nps = np.reshape(nps,(1,48,48))    
    x_test[i] = nps    

np.save('data/x_test.npy',x_test)

x_test = None
"""