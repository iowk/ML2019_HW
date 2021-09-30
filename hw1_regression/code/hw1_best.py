import numpy as np
import sys
import math

class Par():
    def __init__(self,w,b):
        self.w = w
        self.b = b

def norm(a): #2-dim
    t_a = a.transpose() #162*384(192)
    na = np.zeros((a.shape[0],a.shape[1])) #384(192)*162
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            na[i][j] = (a[i][j]-np.average(t_a[j]))/np.std(t_a[j])
    return na

def calc_lsq(x,y):    
    temp = np.ones((x.shape[0],1))
    nx = np.column_stack([x,temp]) #163*384
    tnx = nx.transpose()
    lsq = np.dot(np.dot(np.linalg.inv(np.dot(tnx,nx)),tnx),y)
    return lsq
def grad_des(x,y):
    row = x.shape[0] #384 for train_x
    col = x.shape[1] #162 for train_x
    lsq = calc_lsq(x,y)
    w = lsq[0:col]
    b = lsq[col]
    lsq = None
    itera = 100
    lr = 0.0000001
    lamb = 0
    w_pgrad = np.zeros((col))
    b_pgrad = 0    
    for it in range(itera):
        w_grad = np.zeros((col)) #162
        b_grad = 0       
        for i in range(row):              
            loss = (y[i]-(np.dot(x[i],w)+b))**2+lamb*np.sum((w**2))            
            w_grad = w_grad-2*x[i]*(y[i]-(np.dot(x[i],w)+b))+2*w*lamb
            b_grad = b_grad-2*(y[i]-(np.dot(x[i],w)+b))
        w_pgrad = w_pgrad+w_grad**2
        b_pgrad = b_pgrad+b_grad**2
        w_ada = np.sqrt(w_pgrad)        
        b_ada = math.sqrt(b_pgrad)
        w = w-lr*w_grad/w_ada
        b = b-lr*b_grad/b_ada
    np.save('hw1_best_w.npy',w)
    np.save('hw1_best_b.npy',b)    
    
    
def getWant(x,w):
    nx = np.zeros((x.shape[0],w.shape[0]))    
    for i in range(x.shape[0]):
        for j in range(w.shape[0]):
            nx[i][j] = x[i][w[j]]
    return nx
def fixPM(x): 
    for i in range(x.shape[0]):
        for j in range(9):
            if(x[i][j*18+9] < 0):
                if j==0:
                    x[i][j*18+9] = x[i][27]
                elif j==8:
                    x[i][j*18+9] = x[i][135]
                else:
                    x[i][j*18+9] = (x[i][j*18-9]+x[i][j*18+27])/2
    return x
def caly(par,x):
    y = np.dot(x,par.w)+par.b
    return y

def opt(x,y):
    row = y.shape[0]
    for i in range(row):
        if y[i]-x[i][153]>50:
            y[i] = x[i][153]+50
        if y[i]-x[i][153]<-50:
            y[i] = x[i][153]-50
        if y[i]<0:
            y[i] = 0        
    return y

def calc_err(y_cal,test_y):
    err = test_y-y_cal #192
    return np.sqrt(np.mean((err**2)))

train_x = np.load('train_x.npy')
train_y = np.load('train_y.npy')

#train_x: (5751,162) train_y: (5751)
train_x = fixPM(train_x)
for i in range(train_y.shape[0]):
    if train_y[i]<0:
        train_y[i] = train_x[i][153]
#train_x = np.hstack((train_x,np.ones((train_x.shape[0],1))))
"""
cross_count = 9
n_train_x = np.ones((cross_count,int(5751*(cross_count-1)/cross_count),train_x.shape[1]))
n_train_y = np.ones((cross_count,int(5751*(cross_count-1)/cross_count)))
n_test_x = np.ones((cross_count,int(5751/cross_count),train_x.shape[1]))
n_test_y = np.ones((cross_count,int(5751/cross_count)))
for n in range(cross_count):
    cur_test = 0
    cur_train = 0
    for i in range(0,5751):        
        if i%cross_count==n:                 
            n_test_x[n][cur_test] = train_x[i]            
            n_test_y[n][cur_test] = train_y[i]
            cur_test+=1
        else:                
            n_train_x[n][cur_train] = train_x[i]           
            n_train_y[n][cur_train] = train_y[i] 
            cur_train+=1
    n_test_y[n] = n_test_y[n].flatten()
    n_train_y[n] = n_train_y[n].flatten()
"""
wanted = 0
f_wanted = np.array([5,6,7,8,9,12])
f_wan_len = f_wanted.shape[0]
for i in range(0,9):
    for j in range(f_wan_len): 
        wanted = np.append(wanted,f_wanted[j]+i*18)
wanted = np.delete(wanted,0)


w_train_x = getWant(train_x,wanted)
t_x = w_train_x
t_y = train_y        
grad_des(t_x,t_y)
t_x = None
t_y = None
w_n_train_x = None
w_n_test_x = None

train_x = None
train_y = None
n_train_x = None
n_train_y = None
n_test_x = None
n_test_y = None
