import numpy as np
from keras.layers import Convolution2D,MaxPooling2D,AveragePooling2D,Flatten,Dense,LeakyReLU,BatchNormalization,Dropout,Activation
from keras.models import Sequential,load_model
from keras.callbacks import EarlyStopping
from keras.applications.resnet50 import ResNet50
from keras import optimizers,regularizers
from keras.preprocessing.image import ImageDataGenerator


def train(x,y):
   
    model = Sequential()
    model.add(Convolution2D(filters=64,kernel_size = (3,3),padding = 'same',input_shape=(48,48,1)))    
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Convolution2D(filters=64,kernel_size = (3,3),padding = 'same'))    
    model.add(BatchNormalization())    
    model.add(Activation('relu')) 
    model.add(Convolution2D(filters=64,kernel_size = (3,3),padding = 'same'))    
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Convolution2D(filters=64,kernel_size = (3,3),padding = 'same'))    
    model.add(BatchNormalization())    
    model.add(Activation('relu'))   
    
    model.add(MaxPooling2D((2,2)))
    
    model.add(Convolution2D(filters=128,kernel_size = (3,3),padding = 'same'))    
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Convolution2D(filters=128,kernel_size = (3,3),padding = 'same'))    
    model.add(BatchNormalization())    
    model.add(Activation('relu'))
    model.add(Convolution2D(filters=128,kernel_size = (3,3),padding = 'same'))    
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Convolution2D(filters=128,kernel_size = (3,3),padding = 'same'))    
    model.add(BatchNormalization())    
    model.add(Activation('relu'))    
              
    model.add(MaxPooling2D((2,2)))
              
    model.add(Convolution2D(filters=256,kernel_size = (3,3),padding = 'same'))    
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Convolution2D(filters=256,kernel_size = (3,3),padding = 'same'))    
    model.add(BatchNormalization())    
    model.add(Activation('relu'))
    model.add(Convolution2D(filters=256,kernel_size = (3,3),padding = 'same'))    
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Convolution2D(filters=256,kernel_size = (3,3),padding = 'same'))    
    model.add(BatchNormalization())    
    model.add(Activation('relu'))
    
    model.add(MaxPooling2D((2,2)))
              
    model.add(Convolution2D(filters=512,kernel_size = (3,3),padding = 'same'))    
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Convolution2D(filters=512,kernel_size = (3,3),padding = 'same'))    
    model.add(BatchNormalization())    
    model.add(Activation('relu'))
    model.add(Convolution2D(filters=512,kernel_size = (3,3),padding = 'same'))    
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Convolution2D(filters=512,kernel_size = (3,3),padding = 'same'))    
    model.add(BatchNormalization())    
    model.add(Activation('relu'))
    
    model.add(AveragePooling2D((2,2)))       
    model.add(Flatten())   
    
    model.add(Dense(7,activation='softmax'))
    model.summary()
    Adam = optimizers.adam(lr=0.001)
    model.compile(optimizer=Adam ,loss='categorical_crossentropy',metrics=['accuracy'])
    
    #model = load_model('keras_model.h5')
    x_t = x[0:int(x.shape[0]*9/10)]
    y_t = y[0:int(y.shape[0]*9/10)]
    x_v = x[int(x.shape[0]*9/10):x.shape[0]]
    y_v = y[int(y.shape[0]*9/10):y.shape[0]]
    image_gen = ImageDataGenerator(samplewise_center=True,rotation_range=30,horizontal_flip=True,zoom_range=0.15,width_shift_range=0.15,
    height_shift_range=0.15) 
    val_gen = ImageDataGenerator(samplewise_center=True,rotation_range=30,horizontal_flip=True,zoom_range=0.15,width_shift_range=0.15,
    height_shift_range=0.15)
    image_gen.fit(x_t)
    model.fit_generator(image_gen.flow(x_t,y_t,batch_size=32),epochs=50,validation_data = val_gen.flow(x_v,y_v),verbose=2)
    
    model.save('keras_model.h5')
    

x_train = np.load('x_train.npy').astype(float)
y_train = np.load('y_train.npy')



train(x_train,y_train)

x_train = None
y_train = None