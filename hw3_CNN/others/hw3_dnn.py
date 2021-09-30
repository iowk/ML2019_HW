import numpy as np
from keras.layers import Convolution2D,MaxPooling2D,AveragePooling2D,Flatten,Dense,LeakyReLU,BatchNormalization,Dropout,Activation
from keras.models import Sequential,load_model
from keras.callbacks import EarlyStopping
from keras.applications.resnet50 import ResNet50
from keras import optimizers,regularizers
from keras.preprocessing.image import ImageDataGenerator

def train(x,y):
    model = Sequential()
    
    model.add(Dense(100,activation='relu',input_shape=(48,48,1)))
    model.add(Flatten())
    model.add(Dense(50,activation='relu'))  
    model.add(Dense(7,activation='softmax'))
    model.summary()
    Adam = optimizers.adam(lr=0.001)
    model.compile(optimizer=Adam ,loss='categorical_crossentropy',metrics=['accuracy'])
    x_t = x[0:int(x.shape[0]*9/10)]
    y_t = y[0:int(y.shape[0]*9/10)]
    x_v = x[int(x.shape[0]*9/10):x.shape[0]]
    y_v = y[int(y.shape[0]*9/10):y.shape[0]]
    image_gen = ImageDataGenerator(samplewise_center=True,rotation_range=30,horizontal_flip=True,zoom_range=0.15,width_shift_range=0.15,
    height_shift_range=0.15) 
    val_gen = ImageDataGenerator(samplewise_center=True,rotation_range=30,horizontal_flip=True,zoom_range=0.15,width_shift_range=0.15,
    height_shift_range=0.15)
    image_gen.fit(x_t)
    #val_gen.fit(x_v)
    early_stopping = EarlyStopping(monitor='val_acc',patience=10,verbose=2)
    model.fit_generator(image_gen.flow(x_t,y_t,batch_size=32),epochs=1,validation_data = val_gen.flow(x_v,y_v),verbose=2,callbacks=[early_stopping])
    
    model.save('keras_model_dnn.h5')

x_train = np.load('x_train.npy').astype(float)
y_train = np.load('y_train.npy')
train(x_train,y_train)

x_train = None
y_train = None