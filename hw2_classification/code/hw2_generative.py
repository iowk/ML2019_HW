import numpy as np
import math
import sys


def prob(mu,sigma,x):
    fsiz = x.shape[0]
    delt = np.reshape(x-mu,(fsiz,1))
    #print(np.linalg.det(sigma))
    #print(np.dot(np.transpose(delt),np.linalg.pinv(sigma)))
    val =1/((2*math.pi)**mu.shape[0])/(abs(np.linalg.det(sigma))**(0.5))*math.exp(-0.5*np.dot(np.dot(np.transpose(delt),np.linalg.pinv(sigma)),delt)/10000)
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
x_train_0 = np.load('data/x_train_0.npy')
x_train_1 = np.load('data/x_train_1.npy')
x_test = np.load('data/x_test.npy')
p0 = x_train_0.shape[0]
p1 = x_train_1.shape[0]
print(x_train_0.shape,x_train_1.shape,x_test.shape)

norm_list = np.arange(x_train_0.shape[1], dtype = np.int)
"""
x_train_0 = norm(x_train_0,norm_list)
x_train_1 = norm(x_train_1,norm_list)
"""
x_test = norm(x_test,norm_list)

fsiz = x_train_0.shape[1]

mu_0 = np.array([])
for i in range(fsiz):
    mu_0 = np.append(mu_0,np.mean(x_train_0[0:x_train_0.shape[0],i]))
sigma_0 = np.zeros((fsiz,fsiz))
for i in range(x_train_0.shape[0]):
    delt = np.reshape(x_train_0[i]-mu_0,(fsiz,1))
    sigma_0 = sigma_0 + np.dot(delt,np.transpose(delt))
sigma_0 = sigma_0/x_train_0.shape[0]
mu_1 = np.array([])
for i in range(fsiz):
    mu_1 = np.append(mu_1,np.mean(x_train_1[0:x_train_1.shape[0],i]))
sigma_1 = np.zeros((fsiz,fsiz))
for i in range(x_train_1.shape[0]):
    delt = np.reshape(x_train_1[i]-mu_1,(fsiz,1))
    sigma_1 = sigma_1 + np.dot(delt,np.transpose(delt))
sigma_1 = sigma_1/x_train_1.shape[0]
sigma_co = (x_train_0.shape[0]*sigma_0+x_train_1.shape[0]*sigma_1)/(x_train_0.shape[0]+x_train_1.shape[0]) 
np.save('data/p0.npy',p0)
np.save('data/p1.npy',p1)
np.save('data/mu_0.npy',mu_0)
np.save('data/mu_1.npy',mu_1)
np.save('data/sigma_co.npy',sigma_co)
"""
y_test = np.array([],dtype = np.int)
for i in range(x_test.shape[0]):
    llh_0 = prob(mu_0,sigma_co,x_test[i])*p0
    llh_1 = prob(mu_1,sigma_co,x_test[i])*p1
    #print(llh_0,llh_1)
    if llh_0 > llh_1:
        y_test = np.append(y_test,0)
    else:
        y_test = np.append(y_test,1)
f = open('ans.csv','w')
f.write("id,label"+'\n')
for i in range(y_test.shape[0]):
    msg = str(i+1) + "," + str(y_test[i]) + '\n'
    f.write(msg)
"""

x_train_0 = None
x_train_1 = None
x_test = None
y_test = None
mu_0 = None
mu_1 = None
sigma_0 = None
sigma_1 = None
sigma_co = None
delt = None