#coding=utf-8
import numpy as np
import sys
import jieba
from gensim.models import Word2Vec,word2vec
from keras.models import Sequential,load_model,Model
from keras.layers.core import Dense
from keras.layers.merge import concatenate
from keras.layers import Dropout,LSTM,GRU,CuDNNGRU,TimeDistributed,Activation,Input,GlobalMaxPooling1D,GlobalAveragePooling1D,Lambda,Bidirectional,SpatialDropout1D
from keras import regularizers,optimizers

def av_rnn():
    input_layer = Input(shape=(100,300))
    dropout_layer = SpatialDropout1D(0.25)(input_layer)
    rnn1 = Bidirectional(CuDNNGRU(64,return_sequences=True))(dropout_layer)
    rnn2 = Bidirectional(CuDNNGRU(64,return_sequences=True))(rnn1)
    x = concatenate([rnn1,rnn2],axis=2)  
    last = Lambda(lambda t: t[:, -1], name='last')(x)
    maxpool = GlobalMaxPooling1D()(x)    
    average = GlobalAveragePooling1D()(x)
    all_views = concatenate([last, maxpool, average], axis=1)
    x = Dropout(0.5)(all_views)
    x = Dense(144, activation="relu")(x)
    output_layer = Dense(1, activation="sigmoid")(x)
    model = Model(inputs=input_layer, outputs=output_layer)
    adam_optimizer = optimizers.Adam(lr=1e-3, decay=1e-6, clipvalue=5)
    model.compile(loss='binary_crossentropy', optimizer=adam_optimizer, metrics=['accuracy'])
    model.summary()
    return model

f = open('data/seg_list_all.txt','w',encoding = 'UTF-8')
f1 =  open('data/seg_list.txt','r',encoding = 'UTF-8')
f2 =  open('data/seg_list_test.txt','r',encoding = 'UTF-8')

while True:
    s0 = f1.readline()
    if len(s0) ==0:
        break
    f.write(s0)
while True:
    s0 = f2.readline()
    if len(s0) ==0:
        break
    f.write(s0)
sentences = word2vec.LineSentence('data/seg_list_all.txt')
w2vmodel = word2vec.Word2Vec(sentences,size=300,window=5,min_count=2,workers=8,iter=80,negative=10)
w2vmodel.save('w2v.model')


w2vm = Word2Vec.load('w2v.model')
f = open('data/seg_list.txt','r',encoding = 'UTF-8')
x_seg = []
while True:
    s0 = f.readline()    
    if len(s0) ==0:
        break
    seg_list = s0.split(sep=' ')
    x_seg.append(seg_list)

x_seg = x_seg[:-1]
n = len(x_seg)
print(n)
print(w2vm)
batch_size = 2000
MAX_LENGTH = 100

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
    out_path = 'data/x_inp_w2v/' + str(i) + '.npy'    
    np.save(out_path,x_inp)

model = av_rnn()
#model = load_model('w2v_model.h5')
y_train = np.load('data/y_train.npy')
n_t = n*9/10
epochs = 5
best_acc = 0
acc_his = np.array([])
loss_his = np.array([])
for it in range(epochs):
    print("Epoch:",it)
    for i in range(int(n_t/batch_size)):
        print("Batch num:",i)
        st = i*batch_size
        ed = min((i+1)*batch_size,n_t)
        inp_path = 'data/x_inp_w2v/' + str(i) + '.npy'
        x_inp = np.load(inp_path)
        history = model.fit(x_inp,y_train[st:ed],verbose=2)  
        acc_his = np.append(acc_his,history.history['acc'][0])
        loss_his = np.append(loss_his,history.history['loss'][0])
    cor_tot = 0
    val_tot = 0
    for i in range(int(n_t/batch_size),int(n/batch_size)-1):
        st = i*batch_size
        ed = (i+1)*batch_size
        inp_path = 'data/x_inp_w2v/' + str(i) + '.npy'
        x_inp = np.load(inp_path)
        y_p = model.predict(x_inp)
        for j in range(st,ed):
            val_tot+=1
            if y_train[j][0]==0 and y_p[j-st][0]<0.5: cor_tot+=1
            if y_train[j][0]==1 and y_p[j-st][0]>=0.5: cor_tot+=1
    acc = cor_tot/val_tot
    print("val_acc =",acc)
    if acc > best_acc:
        best_acc = acc
        model.save('w2v_model.h5',model)
x_seg = None
x_inp = None
y_train = None    
