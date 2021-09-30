import sys
import numpy as np
from keras.models import Sequential,load_model,Model
from keras.layers.core import Dense
from keras.layers.merge import concatenate
from keras.layers import Dropout,Activation,Conv2D,MaxPooling2D,UpSampling2D,BatchNormalization
from keras import regularizers,optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from skimage import io
from sklearn.cluster import KMeans
from MulticoreTSNE import MulticoreTSNE as TSNE
from sklearn.decomposition import PCA

in_folder = sys.argv[1]

x = np.zeros((32,32,32,3))
for i in range(1,33):
    if i < 10: in_path = in_folder + '/00000' + str(i) + '.jpg'
    elif i < 100: in_path = in_folder + '/0000' + str(i) + '.jpg'
    x[i-1] = io.imread(in_path)

x = x.astype('float32')/255
model = load_model('keras_model.hdf5')
y = model.predict(x)
x = (x*255).astype('uint8')
y = (y*255).astype('uint8')
for i in range(32):
    o1 = 'original/' + str(i) + '.jpg'
    o2 = 'encoded/' + str(i) + '.jpg'
    io.imsave(o1,x[i])
    io.imsave(o2,y[i])

x = None
y = None