import numpy as np
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
import sys,wget
from PIL import Image

def inpic(n1,n2):
    if n1<0 or n1>47:
        return False
    if n2<0 or n2>47:
        return False
    return True

wget.download('https://www.dropbox.com/s/etnwunuyg6p8kwx/keras_model.h5?dl=1')
#model.save('keras_model.h5',model)
model = load_model('keras_model.h5')
inp_lay = model.layers[0].input
out_lay = model.layers[-1].output
gen = ImageDataGenerator(samplewise_center=True)

x_draw = np.zeros((7,48,48,1))
sal_map = np.ones((7,48,48))


x_train = np.load('x_train.npy')
y_train = np.load('y_train.npy')

print("Getting x_draw")
for i in range(7):
    for j in range(x_train.shape[0]):
        y_temp = model.predict_generator(gen.flow(x_train[j].reshape(1,48,48,1),shuffle=False))
        if y_temp[0][i]>0.9 and y_train[j][i]==1:
            x_draw[i] = x_train[i].copy()
            print("Emotion",i,":",j,"value =",y_temp[0][i])
            break
    

np.save('x_draw.npy',x_draw)
x_draw = np.load('x_draw.npy')
heat_mask = 215
"""
for emo in range(7):
    img = Image.fromarray((x_draw[emo].reshape(48,48)*255).astype('uint8'))
    out_path = "images/ori1_" + str(emo) + ".jpg"
    img.save(out_path) 
"""
for emo in range(7):  
    x_draw[emo] -= np.average(x_draw[emo])
    cost = out_lay[0,emo]
    get_grad = K.function([inp_lay,K.learning_phase()],[K.gradients(cost,inp_lay)[0]])    
    grad = np.absolute(get_grad([x_draw[emo].reshape(1,48,48,1),0]))
    sal_map[emo] = (255-grad*255/grad.max()).reshape(48,48)
    for i in range(48):
        for j in range(48):
            if sal_map[emo][i][j]>heat_mask:
                sal_map[emo][i][j]=255
    img = Image.fromarray(sal_map[emo].astype('uint8'))
    out_path = sys.argv[1] + "/fig1_" + str(emo) + ".jpg"
    img.save(out_path)  
#np.save('sal_map.npy',sal_map)

grad = None
x_draw = None
x_train = None
y_train = None
y_temp = None
y_ori = None
y_mask = None
x_cur = None