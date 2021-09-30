import numpy as np
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
import wget,sys

wget.download("https://www.dropbox.com/s/etnwunuyg6p8kwx/keras_model.h5?dl=1")
model = load_model('keras_model.h5')
x_test = np.load('x_test.npy').astype(float)
x_test = np.reshape(x_test,(x_test.shape[0],48,48,1))/255
test_gen = ImageDataGenerator(samplewise_center=True)
y_test = model.predict_generator(test_gen.flow(x_test,shuffle=False))
y_ans = np.zeros((y_test.shape[0]))
for i in range(y_test.shape[0]):
    if i%1000==0:
        print(i) 
    y_ans[i] = int(np.argmax(y_test[i]))
y_ans.astype(int)
f = open(sys.argv[1],'w')
f.write("id,label"+'\n')
for i in range(y_ans.shape[0]):
    msg = str(i) + "," + str(int(y_ans[i])) + '\n'
    f.write(msg)
x_test = None
y_test = None
y_ans = None
