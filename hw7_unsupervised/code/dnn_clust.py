import numpy as np
from keras.models import Sequential,load_model,Model
from keras.layers.core import Dense
from keras.layers.merge import concatenate
from keras.layers import Dropout,Activation,Conv2D,MaxPooling2D,UpSampling2D
from keras import regularizers,optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from skimage import io
from sklearn.cluster import KMeans
from sklearn import metrics
from MulticoreTSNE import MulticoreTSNE as TSNE
from sklearn.decomposition import PCA

"""
in_folder = 'images/'

x = np.zeros((40000,32,32,3))
for i in range(1,40001):
    if i < 10: in_path = in_folder + '00000' + str(i) + '.jpg'
    elif i < 100: in_path = in_folder + '0000' + str(i) + '.jpg'
    elif i < 1000: in_path = in_folder + '000' + str(i) + '.jpg'
    elif i < 10000: in_path = in_folder + '00' + str(i) + '.jpg'
    else: in_path = in_folder + '0' + str(i) + '.jpg'
    x[i-1] = io.imread(in_path)


x = x.astype('float32')/255

model = load_model('keras_model.hdf5')
temp = (model.predict(x[0].reshape(1,32,32,3))*255).astype('uint8')
io.imsave('test_image.jpg',temp[0])
"""
x = np.load('data/x.npy').astype('float32')/255
x = np.reshape(x,(40000,32*32*3))
x_train = x[0:36000]
x_val = x[36000:40000]
#valgen = ImageDataGenerator(featurewise_center=True,featurewise_std_normalization=True)

model = Sequential()
model.add(Dense(784,activation='relu',input_shape=(32*32*3,)))
model.add(Dense(512,activation='relu'))
model.add(Dense(256,activation='relu'))

model.add(Dense(256,activation='relu'))
model.add(Dense(512,activation='relu'))
model.add(Dense(784,activation='relu'))
model.add(Dense(32*32*3,activation='sigmoid'))
#adam = optimizers.Adam(lr=1e-6)
model.compile(optimizer='adadelta',loss='binary_crossentropy')
model.summary()

#model = load_model('keras_model.hdf5')
filepath = "models/weights.{epoch:02d}-{val_loss:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath,monitor='val_loss',save_best_only=True,mode='min')
model.fit(x_train,x_train,batch_size=512,epochs=50,validation_data=(x_val,x_val),shuffle=True)
model.save('dnn_model.hdf5')
compressed_layer=2
get_comp_output = K.function([model.layers[0].input],[model.layers[compressed_layer].output])
compressed = np.zeros((40000,256))
for i in range(10):
    st = i*4000
    ed = (i+1)*4000
    compressed[st:ed] = get_comp_output([x[st:ed]])[0]
print(compressed.shape)

pca = PCA(n_components=60,copy=False,whiten=True,random_state=0).fit_transform(compressed)
tsne = TSNE(n_components=2,n_jobs=20,random_state=0).fit_transform(pca)
kmeans = KMeans(n_clusters=2,random_state=0,max_iter=2000).fit(tsne)
labels = kmeans.labels_
np.save('data/labels.npy',labels)

x = None
x_train = None
x_val = None
compressed = None
labels = None
