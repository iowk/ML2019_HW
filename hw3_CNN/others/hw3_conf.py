import numpy as np
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
import wget,sys


model = load_model('keras_model.h5')

x_train = np.load('x_train.npy').astype(float)
y_train = np.load('y_train.npy').astype(float)
x_train = np.reshape(x_train,(x_train.shape[0],48,48,1))
x_val = x_train[int(x_train.shape[0]*9/10):x_train.shape[0]]
y_val = y_train[int(y_train.shape[0]*9/10):y_train.shape[0]]

print(y_val.shape[0])

conf_mat = np.zeros((7,7))

test_gen = ImageDataGenerator(samplewise_center=True)

y_pred = model.predict_generator(test_gen.flow(x_val,shuffle=False))

for i in range(y_pred.shape[0]):
    pr = np.argmax(y_pred[i])
    ac = np.argmax(y_val[i])
    conf_mat[pr][ac] += 1.0

conf_mat /= y_val.shape[0]
print(conf_mat)
for i in range(7):
    s = np.sum(conf_mat[i])
    for j in range(7):
        conf_mat[i][j] = conf_mat[i][j]/s
print(conf_mat)

np.save('conf_mat.npy',conf_mat)

x_train = None
y_train = None
x_val = None
y_val = None
y_pred = None
conf_max = None

