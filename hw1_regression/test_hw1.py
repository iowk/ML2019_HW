import numpy as np
import sys


def opt(x,y):
    row = x.shape[0]
    for i in range(row):
        if y[i]-x[i][153]>50:
            y[i] = x[i][153]+50
        if y[i]-x[i][153]<-50:
            y[i] = x[i][153]-50
        if y[i]<0:
            y[i] = 0
    return y

def getWant(x,w):
    nx = np.zeros((x.shape[0],w.shape[0]))    
    for i in range(x.shape[0]):
        for j in range(w.shape[0]):
            nx[i][j] = x[i][w[j]]
    return nx

test_loc = sys.argv[1]
#print(test_loc)
test_x = np.zeros((240,162))
i = 0
c = 0
with open(test_loc,'r',errors = 'ignore') as f:
    for cur in range(240):
        for i in range(18):            
            ts = f.readline()            
            tls = ts.split(sep = ',')
            #print(tls)            
            for j in range(9):
                if len(tls[j+2])>=2:
                    if tls[j+2][0:2] == "NR":
                        tls[j+2] = 0.0
                test_x[cur][j*18+i] = float(tls[j+2])        
#print(test_x.shape)
w = np.load('hw1_w.npy')
b = np.load('hw1_b.npy')
wanted = 0
f_wanted = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17])
f_wan_len = f_wanted.shape[0]
for i in range(0,9):
    for j in range(f_wan_len): 
        wanted = np.append(wanted,f_wanted[j]+i*18)
wanted = np.delete(wanted,0)
w_test_x = getWant(test_x,wanted)

test_y = np.dot(w_test_x,w)+b
#print(test_y.shape)
#print(test_y[0:20])

test_y = opt(test_x,test_y)

out_loc = sys.argv[2]
f = open(out_loc,'w')
f.write("id,value\n")
for i in range(test_y.size):
    out_text = "id_" + str(i) + "," + str(test_y[i]) + "\n"
    f.write(out_text)
test_x = None
w_test_x = None
test_y = None