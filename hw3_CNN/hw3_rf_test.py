import numpy as np
import sys

x_test = np.array([])
f = open(sys.argv[1],'r')
s = f.readline()
while True:
    s0 = f.readline()
    if len(s0) ==0:
        break
    ls0 = s0.split(sep=',')
    ls = ls0[1].split(sep=' ')
    nps = np.array([])
    nps = np.asarray(ls,dtype=np.int)   
    nps = np.reshape(nps,(1,48,48))
    if x_test.size == 0:
        x_test = nps
    else:
        x_test = np.vstack((x_test,nps))
    if x_test.shape[0]%2000==0:
        print(x_test.shape[0])
x_test = np.reshape(x_test,(x_test.shape[0],48,48,1))
np.save('x_test.npy',x_test)
x_test = None