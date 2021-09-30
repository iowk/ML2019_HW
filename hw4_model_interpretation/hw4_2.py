import numpy as np
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
import sys
from PIL import Image


model = load_model('keras_model.h5')

inp_lay = model.inputs[0]
gen = ImageDataGenerator(samplewise_center=True)

num = 3050

x_train = np.load('x_train.npy')
x = x_train[num].copy()
x_train = None

x_ave = np.average(x)
x -= np.average(x)
full_img_1 = Image.new('L',(424,424))
full_img_2 = Image.new('L',(424,424))
for i in range(64):
    if i%21==0:
        print("Processing filter",i)
    
    out_lay = model.layers[4].output
    cost = K.mean(out_lay[:,:,:,i])      
    grad = K.gradients(cost,inp_lay)[0]
    grad = grad/(K.sqrt(K.mean(K.square(grad)))+1e-5)
    get_grad = K.function([inp_lay],[grad])
    
    x_asc = np.ones((1,1,48,48,1))/10
    epochs = 20
    lr = 0.1
    for it in range(epochs):
        x_grad = get_grad([x_asc.reshape(1,48,48,1)])
        x_asc += np.multiply(x_grad,lr)
    x_asc = x_asc.reshape(48,48)    
    x_asc = x_asc-x_asc.min() 
    x_asc = x_asc*255/(x_asc.max()+1e-5)
    img = Image.fromarray(x_asc.astype('uint8').reshape(48,48),mode='L')    
    col = i%8
    row = int(i/8)
    full_img_1.paste(img,(5+col*48+(col-1)*5,5+row*48+(row-1)*5))
    
    out_lay = model.layers[4].output
    cost = K.mean(out_lay[:,:,:,i])      
    grad = K.gradients(cost,inp_lay)[0]
    grad = grad/(K.sqrt(K.mean(K.square(grad)))+1e-5)
    get_grad = K.function([inp_lay],[grad])
    x_asc = x.copy().reshape(1,1,48,48,1)
    epochs = 20
    lr = 0.1
    for it in range(epochs):
        x_grad = get_grad([x_asc.reshape(1,48,48,1)])        
        x_asc += np.multiply(x_grad,lr)
    x_asc = x_asc.reshape(48,48)
    x_asc = (x_asc+x_ave)*255
    img = Image.fromarray(x_asc.astype('uint8').reshape(48,48))
    col = i%8
    row = int(i/8)
    full_img_2.paste(img,(5+col*48+(col-1)*5,5+row*48+(row-1)*5))
    
 
out_path = sys.argv[1] + "/fig2_1.jpg"
full_img_1.save(out_path)

out_path = sys.argv[1] + "/fig2_2.jpg"
full_img_2.save(out_path)

x = None
x_asc = None