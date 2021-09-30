#coding=utf-8
import numpy as np
import sys
import jieba
from gensim.models import Word2Vec
from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers import Dropout
from keras import regularizers


jieba.set_dictionary(sys.argv[2])
f = open(sys.argv[1],'r',encoding = 'UTF-8')
s = f.readline()
out_path = 'data/seg_list.txt'
out = open(out_path,'w',encoding = 'UTF-8')
while True:
    s0 = f.readline()
    #if len(x_seg)>100: break
    if len(s0) ==0:
        break
    sent = s0.split(sep=',')[1]
    seg_list = jieba.lcut(sent)
    for item in seg_list:        
        out.write(item + ' ')    
    seg_list = None
    
    
