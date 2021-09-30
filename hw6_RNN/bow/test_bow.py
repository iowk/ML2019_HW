#coding=utf-8
import numpy as np
import sys
import jieba
from keras.models import load_model

jieba.set_dictionary('data/dict.txt.big')

f = open('data/wlist.csv','r',encoding = 'UTF-8')
wlist = []
while True:
    s0 = f.readline()
    s0.rstrip('\n')
    if len(s0)==0:
        break
    wlist.append(s0[0:-1])
lw = len(wlist)-1
dict1 = {}

for item in wlist:
    if not item in dict1:
        dict1[item] = 0

f = open('data/seg_list_test.txt','r',encoding = 'UTF-8')


n=20000
batch_size = 2000
y_test = np.zeros((n))
for i in range(int(n/batch_size)): 
    print("Batch num:",i)
    st = i*batch_size
    ed = min((i+1)*batch_size,n)     
    x_t = np.zeros((ed-st,lw))
    for j in range(st,ed):
        dict1 = dict.fromkeys(dict1,0)
        for item in x_seg[j]:
            if item in dict1:
                dict1[item] +=1 
        x_t[j-st] = np.array(list(dict1.values()))  
    x_t.astype('int8')        
    out_path = 'data/x_test/x_t' + str(i) + '.npy' 
    np.save(out_path,x_t)
    
    in_path = 'data/x_test/x_t' + str(i) + '.npy'         
    x_t = np.load(in_path)
       
    y_p = model.predict(x_t) 
    tot = 0
    for j in range(st,ed):
        if y_p[j-st][0] >= 0.5:  
            tot+=1
            y_test[j] = 1  
    print(tot)

f = open(sys.argv[1],'w')
f.write("id,label"+'\n')
for i in range(y_test.shape[0]):
    msg = str(i) + "," + str(int(y_test[i])) + '\n'
    f.write(msg)

y_test = None
y_p = None
x_t = None
x_seg = None
dict1 = None
wlist = None

    