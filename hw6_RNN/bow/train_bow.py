#coding=utf-8
import numpy as np
import sys
import jieba
from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers import Dropout
from keras import regularizers
from keras.models import load_model

jieba.set_dictionary('data/dict.txt.big')
f = open('data/seg_list.txt','r',encoding = 'UTF-8')

x_seg = []
dict1 = {}
while True:
    s0 = f.readline()    
    if len(s0) ==0:
        break
    seg_list = s0.split(sep=' ')
    x_seg.append(seg_list)
    for item in seg_list:
        if not item in dict1:
            dict1[item] = 0
x_seg = x_seg[:-1]
#print("Keys:", dict1.keys())
wlist = list(dict1.keys())
"""
with open('data/wlist.csv','w',encoding = 'UTF-8') as f:
    for item in wlist:
        f.write("%s\n"%item)
"""
print(len(wlist))
print(len(x_seg))

n = len(x_seg)
n_t = int(len(x_seg)*9/10)
lw = len(wlist)
y_train = np.load('data/y_train.npy')

model = Sequential()
model.add(Dense(100,input_dim=lw,activation = 'relu'))
model.add(Dense(10,activation = 'relu'))
model.add(Dense(10,activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(1,activation = 'sigmoid'))
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
model.summary()


#model = load_model('keras_model.h5')  
epoch = 5
batch_size = 2000
best_acc = 0
acc_his = np.array([])
loss_his = np.array([])
for it in range(epoch):   
    print("Epoch:",it)
    for i in range(int(n_t/batch_size)):
        print("Batch num:",i)        
        st = i*batch_size
        ed = min((i+1)*batch_size,n_t)
        
        if it==0:
            x_t = np.zeros((ed-st,lw))
            for j in range(st,ed):
                dict1 = dict.fromkeys(dict1,0)
                for item in x_seg[j]:
                    dict1[item] +=1 
                x_t[j-st] = np.array(list(dict1.values()))  
            x_t.astype('int8')        
            out_path = 'data/x_t/x_t' + str(i) + '.npy' 
            np.save(out_path,x_t)
        
        in_path = 'data/x_t/x_t' + str(i) + '.npy'         
        x_t = np.load(in_path)
        history = model.fit(x_t,y_train[st:ed],verbose=2) 
        acc_his = np.append(acc_his,history.history['acc'][0])
        loss_his = np.append(loss_his,history.history['loss'][0])
    
    cor_tot = 0
    val_tot = 0
    for i in range(int(n_t/batch_size),int(n/batch_size)-1):
        print("Batch num:",i)
        
        st = i*batch_size
        ed = min((i+1)*batch_size,n)  
        if it==0:
            x_t = np.zeros((ed-st,lw))
            for j in range(st,ed):
                dict1 = dict.fromkeys(dict1,0)
                for item in x_seg[j]:
                    dict1[item] +=1 
                x_t[j-st] = np.array(list(dict1.values()))  
            x_t.astype('int8')        
            out_path = 'data/x_t/x_t' + str(i) + '.npy' 
            np.save(out_path,x_t)
        
        in_path = 'data/x_t/x_t' + str(i) + '.npy'         
        x_t = np.load(in_path)
               
        y_p = model.predict(x_t) 
        for j in range(st,ed):
            val_tot+=1
            if y_p[j-st][0] >= 0.5 and y_train[j][0]==1:
                cor_tot += 1
            if y_p[j-st][0] < 0.5 and y_train[j][0]==0:
                cor_tot += 1
    acc = cor_tot/val_tot
    print("val_acc =",acc)
    if acc > best_acc:
        best_acc = acc
        model.save('keras_model.h5',model)
np.save('acc_his.npy',acc_his)
np.save('loss_his.npy',loss_his) 

"""
f = open(sys.argv[2],'r',encoding = 'UTF-8')
s = f.readline()
x_seg = []
while True:
    s0 = f.readline()
    #if len(x_seg)>100: break
    if len(s0) ==0:
        break
    sent = s0.split(sep=',')[1]
    seg_list = jieba.lcut(sent)    
    x_seg.append(seg_list)

model = load_model('keras_model.h5')         
n=20000
batch_size = 2000
y_test = np.zeros((n))
for i in range(int(n/batch_size)): 
    print("Batch num:",i)
    st = i*batch_size
    ed = min((i+1)*batch_size,n)        
    in_path = 'data/x_test/x_t' + str(i) + '.npy'         
    x_t = np.load(in_path)
   
    y_p = model.predict(x_t) 
    tot = 0
    for j in range(st,ed):
        if y_p[j-st][0] >= 0.5:  
            tot+=1
            y_test[j] = 1  
    print(tot)

f = open(sys.argv[3],'w')
f.write("id,label"+'\n')
for i in range(y_test.shape[0]):
    msg = str(i) + "," + str(int(y_test[i])) + '\n'
    f.write(msg)
"""

y_train =  None
y_test = None
y_p = None
x_t = None
x_seg = None
dict1 = None
wlist = None
