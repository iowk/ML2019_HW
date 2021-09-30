import numpy as np
import math
import sys

def sigmoid(x):
    val = 1/(math.exp(-x)+1)
    #if val<=0 or val>=1:
        #print("error in sigmoid",x,val)
    return val
def cross_entropy(w,x,y):    
    val = 0
    for i in range(x.shape[0]):
        f_x = sigmoid(np.sum(np.dot(x[i],w)))
        if f_x!=0 or f_x!=1:
            val = val-(y[i][0]*math.log(f_x)+(1-y[i][0]*math.log(1-f_x)))        
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
def train(x,y):
    dnum = x.shape[0]
    fnum = x.shape[1]
    w = np.load('data/w.npy')
    #w = np.ones((fnum,1))/100
    lr = 0.01
    epochs = 31
    w_pgrad = np.ones((fnum,1))/100000000
    lamb=0.0001
    for it in range(epochs):
        w_grad = np.zeros((fnum,1))   
        sum_loss = 0
        for i in range(dnum):
            sig = sigmoid(np.sum(np.dot(x[i],w)))
            #loss = -cross_entropy(w,x,y)
            loss = y[i][0]-sig
            sum_loss+=loss
            w_grad = w_grad-2*(np.reshape(x[i],(fnum,1))*(y[i][0]-sig))+2*w*lamb           
        w_pgrad+=w_grad**2
        ada = np.sqrt(w_pgrad)
        w = w-lr*w_grad/ada
        if it%10 == 0:
            print("Loss %3d"%it,"=",sum_loss)
    return w
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
def validate(y_cal,y_test):
    cor = 0
    for i in range(y_cal.shape[0]):
        if y_cal[i] == y_test[i]:
            cor+=1
    return cor/y_cal.shape[0]
    
x_train = np.load('data/x_train.npy')
y_train = np.load('data/y_train.npy')
x_test = np.load('data/x_test.npy')

norm_list = np.array([0,1,3,4,5], dtype = np.int)
x_train = norm(x_train,norm_list)
x_test = norm(x_test,norm_list)

x_train = np.delete(x_train,1,1)
x_test = np.delete(x_test,1,1)
"""
print("Normalize finished")

cc = 5
tot_acc = 0
for n in range(cc):
    val_num = np.array([],dtype = np.int)    
    x_val_c = np.array([])
    y_val_c = np.array([])
    for i in range(x_train.shape[0]):
        if i%cc == n: 
            val_num = np.append(val_num,i)
            if x_val_c.size == 0:
                x_val_c = x_train[i]
                y_val_c = y_train[i]                
            else:
                x_val_c = np.row_stack((x_val_c,x_train[i]))
                y_val_c = np.row_stack((y_val_c,y_train[i]))  
    x_train_c = np.delete(x_train,val_num,0)
    y_train_c = np.delete(y_train,val_num,0)
    x_train_c = np.column_stack((x_train_c,np.ones((x_train_c.shape[0],1))))
    x_val_c = np.column_stack((x_val_c,np.ones((x_val_c.shape[0],1))))
    print("Start training",n+1)
    w = train(x_train_c,y_train_c)
    print("Start validating",n+1)
    y_cal = caly(w,x_val_c)
    acc = validate(y_cal,y_val_c)
    print(acc)
    tot_acc+=acc
    x_train_c = None
    y_train_c = None
    x_val_c = None
    y_cal = None
    y_val_c = None
    w = None
tot_acc = tot_acc/cc
print(tot_acc)
x_train = None
y_train = None
x_test = None
"""
x_train = np.column_stack((x_train,np.ones((x_train.shape[0],1))))
print("Start training")
w = train(x_train,y_train)
np.save('data/w.npy',w)
w = np.load('data/w.npy')
x_test = np.column_stack((x_test,np.ones((x_test.shape[0],1))))
y_test = caly(w,x_test)
f = open('ans_logistsic.csv','w')
f.write("id,label"+'\n')
for i in range(y_test.shape[0]):
    msg = str(i+1) + "," + str(y_test[i][0]) + '\n'
    f.write(msg)

x_train = None
y_train = None
x_test = None
y_test = None
