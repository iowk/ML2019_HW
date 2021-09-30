import numpy as np
import math
import sys

def sigmoid(x):
    val = 1/(math.exp(-x)+1)    
    return val
def norm(x,n):
    for i in range(n.shape[0]):
        x_mean = np.mean(x[0:x.shape[0],n[i]])
        x_std = np.std(x[0:x.shape[0],n[i]])
        if x_std==0:
            x[0:x.shape[0],n[i]] = np.zeros((x.shape[0]))
        else:
            x[0:x.shape[0],n[i]] =  (x[0:x.shape[0],n[i]]-x_mean)/x_std
    return x
def caly(w,x):
    dnum = x.shape[0]
    y = np.zeros((dnum,1),dtype = np.int)
    for i in range(dnum):
        sig = sigmoid(np.dot(x[i],w))
        if sig>0.5:
            y[i][0] = 1
        else:
            y[i][0] = 0
    return y
x_test = np.array([], dtype = np.float64)
f = open(sys.argv[5],mode = 'r')
line = f.readline()
while(True):
    line = f.readline()
    if len(line) == 0:
        break
    line_list = line.split(',')
    line_np = np.array([])
    for i in range(len(line_list)):
        line_np = np.append(line_np,float(line_list[i]))
    if(x_test.size==0):
        x_test = line_np
    else:
        x_test = np.row_stack((x_test,line_np))
norm_list = np.array([0,1,3,4,5], dtype = np.int)
x_test = norm(x_test,norm_list)
x_test = np.delete(x_test,1,1)



w = np.load('data/w.npy')
x_test = np.column_stack((x_test,np.ones((x_test.shape[0],1))))
y_test = caly(w,x_test)
f = open(sys.argv[6],'w')
f.write("id,label"+'\n')
for i in range(y_test.shape[0]):
    msg = str(i+1) + "," + str(y_test[i][0]) + '\n'
    f.write(msg)

x_test = None
y_test = None