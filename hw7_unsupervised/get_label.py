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

x = np.zeros((40000,32,32,3))
for i in range(1,40001):
    if i < 10: in_path = in_folder + '/00000' + str(i) + '.jpg'
    elif i < 100: in_path = in_folder + '/0000' + str(i) + '.jpg'
    elif i < 1000: in_path = in_folder + '/000' + str(i) + '.jpg'
    elif i < 10000: in_path = in_folder + '/00' + str(i) + '.jpg'
    else: in_path = in_folder + '/0' + str(i) + '.jpg'
    x[i-1] = io.imread(in_path)

#x = np.load('data/x.npy')
x = x.astype('float32')/255
model = load_model('keras_model.hdf5')
compressed_layer=8
get_comp_output = K.function([model.layers[0].input],[model.layers[compressed_layer].output])
compressed = np.zeros((40000,4,4,16))
for i in range(10):
    st = i*4000
    ed = (i+1)*4000
    compressed[st:ed] = get_comp_output([x[st:ed]])[0]
print(compressed.shape)
compressed = np.reshape(compressed,(40000,4*4*16))

pca = PCA(n_components=60,copy=False,whiten=True,random_state=0).fit_transform(compressed)
tsne = TSNE(n_components=2,n_jobs=20,random_state=0).fit_transform(pca)
np.save('data/tsne.npy',tsne)
kmeans = KMeans(n_clusters=2,random_state=0,max_iter=2000).fit(tsne)
labels = kmeans.labels_
np.save('data/labels.npy',labels)

x = None