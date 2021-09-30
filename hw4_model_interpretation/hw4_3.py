import numpy as np
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
import sys
from skimage.segmentation import slic
from lime import lime_image
from PIL import Image

model = load_model('keras_model.h5')

def predict(x_rgb):
    #model = load_model('keras_model.h5')
    x = x_rgb[0:x_rgb.shape[0],0:x_rgb.shape[1],0:x_rgb.shape[2],0]
    x = x.reshape(x.shape[0],48,48,1)
    y = model.predict(x)    
    return y

def segmentation(x_rgb):
    return slic(x_rgb)

gen = ImageDataGenerator(samplewise_center=True)

#10,299,9,7,6,15,4
x_draw = np.load('x_draw.npy')

x_draw_rgb = np.zeros((7,48,48,3))
for i in range(7):
    for j in range(3):    
        for k in range(48):
            for l in range(48):
                x_draw_rgb[i][k][l][j] = x_draw[i][k][l][0]

explainer = lime_image.LimeImageExplainer()

for emo in range(7):
    print("Starting task",emo)
    np.random.seed(16)
    explaination = explainer.explain_instance(
            image=x_draw_rgb[emo],top_labels=7,classifier_fn=predict,segmentation_fn=segmentation)
    image,mask = explaination.get_image_and_mask(label=emo,positive_only=False,hide_rest=False,num_features=5,min_weight=0.0)
    image = image*255
    img = Image.fromarray(image.astype('uint8'))
    out_path = sys.argv[1] + "/fig3_" + str(emo) + ".jpg"
    img.save(out_path)

x_draw = None
x_draw_rgb = None
