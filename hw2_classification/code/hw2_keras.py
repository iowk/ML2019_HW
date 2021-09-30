import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense
from keras import regularizers

def train(x,y):
    dnum = x.shape[0]
    fnum = x.shape[1]
    model = Sequential()
    #model.add(Dropout(0.2,input_shape = (fnum,)))
    model.add(Dense(10,input_dim=fnum,activation = 'relu',kernel_regularizer=regularizers.l2(0.001)))  
    model.add(Dense(1,activation = 'sigmoid'))
    model.compile(optimizer='RMSProp',loss='binary_crossentropy',metrics=['accuracy'])
    model.fit(x,y,epochs=100,batch_size=100,verbose=0)
    return model

def norm(x,n):
    for i in range(n.shape[0]):
        x_mean = np.mean(x[0:x.shape[0],n[i]])
        x_std = np.std(x[0:x.shape[0],n[i]])
        if x_std==0:
            x[0:x.shape[0],n[i]] = np.zeros((x.shape[0]))
        else:
            x[0:x.shape[0],n[i]] =  (x[0:x.shape[0],n[i]]-x_mean)/x_std
    return x

x_train = np.load('data/x_train.npy')
y_train = np.load('data/y_train.npy')
x_test = np.load('data/x_test.npy')
#norm_list = np.arange(x_test.shape[1], dtype = np.int)
norm_list = np.array([0,1,3,4,5],dtype = np.int)
#norm_list = np.array([0,1,2,3,4],dtype = np.int)
x_train = norm(x_train,norm_list)
x_test = norm(x_test,norm_list)
#gender at last column
x_train = np.column_stack((x_train,np.ones((x_train.shape[0],1))))
fnum = x_train.shape[1]
for i in range(x_train.shape[0]):
    if x_train[i][2]==1:
        x_train[i][fnum-1] = 0
x_test = np.column_stack((x_test,np.ones((x_test.shape[0],1))))
for i in range(x_test.shape[0]):
    if x_test[i][2]==1:
        x_test[i][fnum-1] = 0
#delete fnlwgt
x_train = np.delete(x_train,1,1)
x_test = np.delete(x_test,1,1)
#bias
x_train = np.column_stack((x_train,np.ones((x_train.shape[0],1))))
x_test = np.column_stack((x_test,np.ones((x_test.shape[0],1))))

print("Normalize finished")
"""
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
    print("Start training",n+1)
    model = train(x_train_c,y_train_c)
    print("Start validating",n+1) 
    loss,acc = model.evaluate(x_val_c,y_val_c)
    print("val_loss =",loss)
    print("val_acc  =",acc)
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
print("Start training")
model = train(x_train,y_train)
model.save('keras_model.h5')
y_test = model.predict(x_test)
for i in range(y_test.shape[0]):
    if(y_test[i][0]>0.5):
        y_test[i][0] = 1
    else:
        y_test[i][0] = 0
y_test = y_test.astype(int)
f = open('ans_best.csv','w')
f.write("id,label"+'\n')
for i in range(y_test.shape[0]):
    msg = str(i+1) + "," + str(y_test[i][0]) + '\n'
    f.write(msg)

x_train = None
y_train = None
x_test = None
y_test = None
