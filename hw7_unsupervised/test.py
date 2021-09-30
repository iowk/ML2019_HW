import numpy as np
import sys

in_path = sys.argv[1]
out_path = sys.argv[2]
f = open(in_path,'r')
out = open(out_path,'w')
out.write('id,label'+'\n')
labels = np.load('data/labels.npy')
s = f.readline()
i = 0
while True:
    s0 = f.readline()    
    if len(s0) ==0:
        break
    n1 = int(s0.split(sep=',')[1])-1
    n2 = int(s0.split(sep=',')[2])-1
    #print(n1,n2)
    if labels[n1]==labels[n2]: out.write(str(i) + ",1" + '\n')
    else: out.write(str(i) + ",0" + '\n')
    i+=1



label = None