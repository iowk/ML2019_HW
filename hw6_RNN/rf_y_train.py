import numpy as np
import sys

y_train = np.array([])
f = open(sys.argv[1],'r')
s = f.readline()
while True:
    s0 = f.readline()
    if len(s0) ==0:
        break
    ls0 = s0.split(sep=',')           
    npy = int(ls0[1][0])   
    y_train = np.append(y_train,npy)   
y_train = np.expand_dims(y_train,1)
print(y_train.shape)
y_train.astype(int)
np.save('data/y_train.npy',y_train)
y_train = None
y_temp = None

    