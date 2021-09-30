#coding=utf-8
import numpy as np
import sys
import jieba
from gensim.models import Word2Vec,word2vec
from keras.models import Sequential,load_model,Model
from keras import regularizers,optimizers


w2vm = Word2Vec.load('w2v.model')


f = open('data/seg_list_test.txt','r',encoding = 'UTF-8')
x_seg = []
while True:
    s0 = f.readline()    
    if len(s0) ==0:
        break
    seg_list = s0.split(sep=' ')
    x_seg.append(seg_list)

x_seg = x_seg[:-1]
n = len(x_seg)
print(w2vm)
print(n)
batch_size = 2000
MAX_LENGTH = 100

y_test = np.zeros((n))
model = load_model('w2v_model.h5')
for i in range(int(n/batch_size)):
    print("Batch num:",i)
    st = i*batch_size
    ed = min((i+1)*batch_size,n)
    x_inp = np.zeros((ed-st,MAX_LENGTH,300))
    for j in range(st,ed):
        for k in range(MAX_LENGTH):
            if k < len(x_seg[j]):
                word = x_seg[j][k]
                if word in w2vm.wv.vocab:
                    x_inp[j-st][k] = w2vm.wv[word]
    y_pred = model.predict(x_inp)
    for j in range(st,ed):
        if y_pred[j-st][0] >= 0.5:
            y_test[j] = 1        

f = open(sys.argv[1],'w')
f.write("id,label"+'\n')
for i in range(y_test.shape[0]):
    msg = str(i) + "," + str(int(y_test[i])) + '\n'
    f.write(msg)
    

y_test = None
y_pred = None
x_seg = None
x_inp = None
y_train = None    
