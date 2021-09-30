import numpy as np
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
import sys
from PIL import Image

def inpic(n1,n2):
    if n1<0 or n1>47:
        return False
    if n2<0 or n2>47:
        return False
    return True

#model = wget.download('https://github.com/iowk3050/ML2019SPRING/releases/download/v0/keras_model.h5')

model = load_model('keras_model.h5')
gen = ImageDataGenerator(samplewise_center=True)

x_draw = np.zeros((7,48,48,1))
sal_map = np.ones((7,48,48))*255

"""
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
"""
x_draw = np.load('x_draw.npy')
mask_size = 11
m = int((mask_size-1)/2)
for emo in range(7):
    print("Starting task",emo)
    x_cur = x_draw[emo].copy()     
    y_ori = model.predict_generator(gen.flow(x_draw[emo].reshape(1,48,48,1),shuffle=False))    
    for i in range(48):
        print("i =",i)
        for j in range(48):
            x_cur = x_draw[emo].copy()
            for k in range(i-m,i+m):
                for l in range(j-m,j+m):
                    if inpic(k,l):
                        x_cur[k][l][0] = 0                       
            y_mask = model.predict_generator(gen.flow(x_cur.reshape(1,48,48,1),shuffle=False))
            if y_ori[0][emo]>y_mask[0][emo]:
                sal_map[emo][i][j] = y_mask[0][emo]/y_ori[0][emo]*255            
    print(sal_map[emo])
    
    img = Image.fromarray(sal_map[emo].astype('uint8'))
    out_path = "images/fig4_" + str(emo) + ".jpg"
    img.save(out_path)  
    img = Image.fromarray((x_draw[emo].reshape(48,48)*255).astype('uint8'))
    out_path = "images/ori4_" + str(emo) + ".jpg"
    img.save(out_path) 
np.save('sal_map.npy',sal_map)


x_draw = None
x_train = None
y_train = None
y_temp = None
y_ori = None
y_mask = None
x_cur = None