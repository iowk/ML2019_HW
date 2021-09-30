import numpy as np
import math
import sys

def prob(mu,sigma,x):
    fsiz = x.shape[0]
    delt = np.reshape(x-mu,(fsiz,1))
    #print(np.linalg.det(sigma))
    v1 = math.log(1/((2*math.pi)**mu.shape[0])/(abs(np.linalg.det(sigma))**(0.5)))
    v2 = -0.5*np.dot(np.dot(np.transpose(delt),np.linalg.pinv(sigma)),delt)    
    val = v1+v2
    delt = None
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

norm_list = np.arange(x_test.shape[1], dtype = np.int)
x_test = norm(x_test,norm_list)
y_test = np.array([],dtype = np.int)
mu_0 = np.load('data/mu_0.npy')
mu_1 = np.load('data/mu_1.npy')
sigma_co = np.load('data/sigma_co.npy')
p0 = np.load('data/p0.npy')
p1 = np.load('data/p1.npy')
for i in range(x_test.shape[0]):
    llh_0 = prob(mu_0,sigma_co,x_test[i])+math.log(p0)
    llh_1 = prob(mu_1,sigma_co,x_test[i])+math.log(p1)
    #print(llh_0,llh_1)
    if llh_0 > llh_1:
        y_test = np.append(y_test,0)
    else:
        y_test = np.append(y_test,1)
f = open(sys.argv[6],'w')
f.write("id,label"+'\n')
for i in range(y_test.shape[0]):
    msg = str(i+1) + "," + str(y_test[i]) + '\n'
    f.write(msg)

x_test = None
y_test = None