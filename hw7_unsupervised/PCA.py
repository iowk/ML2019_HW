import numpy as np
import sys
from skimage import io
from numpy import linalg as LA

in_folder = sys.argv[1]
in_img = sys.argv[2]
out_img = sys.argv[3]

x = np.zeros((415,600,600,3))
for i in range(415):
    in_path = in_folder + str(i) + '.jpg'
    x[i] = io.imread(in_path)
x = np.reshape(x,(415,600*600*3)).astype('float32')
x_ave = np.mean(x,axis=0)
#io.imsave('img_report/avg_face.jpg',np.reshape(x_ave,(600,600,3)).astype(np.uint8))
x = x-x_ave
u,sigma,v = LA.svd(x,full_matrices=False)
idx = sigma.argsort()[::-1]
sigma = sigma[idx]
v = v[idx]
#print(u.shape,sigma.shape,v.shape)

in_path = in_folder + '/' + in_img
x_inp = io.imread(in_path)
x_inp  = np.reshape(x_inp,(600*600*3)).astype('float32')
x_inp = x_inp - x_ave
cur = x_ave
for j in range(5):
    cur = cur+np.dot(x_inp,v[j])*v[j]
cur = np.reshape(cur,(600,600,3))
cur-=np.min(cur)
cur/=np.max(cur)
cur = (cur*255).astype(np.uint8)
out_path = out_img
io.imsave(out_path,cur)
"""
for i in range(10):
    cur = np.reshape(v[i],(600,600,3))
    cur-=np.min(cur)
    cur/=np.max(cur)
    cur = (cur*255).astype(np.uint8)
    out_path = out_folder + str(i) + '.jpg'
    io.imsave(out_path,cur)
"""
cur = None
x = None
x_ave = None
temp = None
sigma = None
u = None
v = None