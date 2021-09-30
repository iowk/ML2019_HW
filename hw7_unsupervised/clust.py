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
from sklearn import metrics
from MulticoreTSNE import MulticoreTSNE as TSNE
from sklearn.decomposition import PCA

in_folder = 'data/images/'

x = np.zeros((40000,32,32,3))
for i in range(1,40001):
    if i < 10: in_path = in_folder + '00000' + str(i) + '.jpg'
    elif i < 100: in_path = in_folder + '0000' + str(i) + '.jpg'
    elif i < 1000: in_path = in_folder + '000' + str(i) + '.jpg'
    elif i < 10000: in_path = in_folder + '00' + str(i) + '.jpg'
    else: in_path = in_folder + '0' + str(i) + '.jpg'
    x[i-1] = io.imread(in_path)


#x = np.load('data/x.npy')
x = x.astype('float32')/255
x_train = x[0:36000]
x_val = x[36000:40000]

#datagen = ImageDataGenerator(rescale=1./255,width_shift_range=0.15,height_shift_range=0.15,                             rotation_range=30,zoom_range=0.15,horizontal_flip=True)
#valgen = ImageDataGenerator(rescale=1./255)

model = Sequential()
"""
model.add(Dense(1000,activation='relu',input_shape=(32*32*3,)))
model.add(Dense(32*32*3,activation='sigmoid'))
"""
model.add(Conv2D(64,(3,3),activation='relu',padding='same',input_shape=(32,32,3)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2),padding='same'))
model.add(Conv2D(32,(3,3),activation='relu',padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2),padding='same'))
model.add(Conv2D(16,(3,3),activation='relu',padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2),padding='same'))

model.add(Conv2D(16,(3,3),activation='relu',padding='same'))
model.add(BatchNormalization())
model.add(UpSampling2D((2,2)))
model.add(Conv2D(32,(3,3),activation='relu',padding='same'))
model.add(BatchNormalization())
model.add(UpSampling2D((2,2)))
model.add(Conv2D(64,(3,3),activation='relu',padding='same'))
model.add(BatchNormalization())
model.add(UpSampling2D((2,2)))
model.add(Conv2D(3,(3,3),padding='same'))
model.add(BatchNormalization())
model.add(Activation('sigmoid'))

#adam = optimizers.Adam(lr=1e-6)
model.compile(optimizer='adadelta',loss='binary_crossentropy')
#model.summary()

model = load_model('keras_model.hdf5')
filepath = "models/weights.{epoch:02d}-{val_loss:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath,monitor='val_loss',save_best_only=True,mode='min')

#train_gen = datagen.flow_from_directory('data/train',target_size=(32,32),color_mode='rgb',                                   batch_size=128,class_mode='input')
#val_gen = valgen.flow_from_directory('data/val',target_size=(32,32),color_mode='rgb',                                   batch_size=128,class_mode='input')
model.fit(x_train,x_train,epochs=20,batch_size=512,validation_data=(x_val,x_val),shuffle=True)
model.save('keras_model.hdf5',model)


x = None
x_train = None
x_val = None
compressed = None
labels = None
