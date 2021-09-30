import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms.functional as TF
from PIL import Image
import sys
from torch.autograd.gradcheck import zero_gradients

epsilon = 20/255

model = torchvision.models.densenet169(pretrained=True)

model.eval()
criterion = nn.CrossEntropyLoss()
mean = torch.tensor([0.485, 0.456, 0.406])
std = torch.tensor([0.229, 0.224, 0.225])

suc_tot = 0

for i in range(200):
    if i<10:
        cnum = "00" + str(i)
    elif i<100:
        cnum = "0" + str(i)
    else:
        cnum = str(i)
    
    image_path = sys.argv[1] + "/" + cnum + ".png"
    img = Image.open(image_path)
    
    x0 = TF.to_tensor(img)
    x0 = torch.transpose(x0,0,2)    
    x0 = torch.transpose(x0,0,1)    
    x0 = (x0 - mean)/std 
    x0 = torch.transpose(x0,0,1)
    x0 = torch.transpose(x0,0,2)
    x0 = x0.unsqueeze(0)
    px0 = x0
    #px0 = preprocess_image(px0)    
    y0 = model(px0)
    true_class = torch.max(y0,1)[1]     
    xstar = x0    
    g_last = 0          
    xstar.requires_grad_(True)
    output = model(xstar)    
    loss = criterion(output,true_class)
    loss.backward()
    g = xstar.grad.data     
    xstar.data += epsilon*torch.sign(g)
    xstar.grad.data.zero_()
    x1 = xstar   
    px1 = x1
    #px1 = preprocess_image(px1)
    y1 = model(px1)
    print("Test",i)
    print("True:",torch.max(y0,1)[1])
    print("Get :",torch.max(y1,1)[1])   
    if torch.max(y0,1)[1]!=torch.max(y1,1)[1]:
        print("Success")
        suc_tot+=1
    else:
        print("Fail")
    out_path = sys.argv[2] + "/" + cnum + ".png"    
    x1 = x1.squeeze()     
    x1 = torch.transpose(x1,0,2)    
    x1 = torch.transpose(x1,0,1)    
    x1 = x1*std+mean
    x1 = torch.transpose(x1,0,1)
    x1 = torch.transpose(x1,0,2)    
    torchvision.utils.save_image(x1,out_path)
x0 = None
x1 = None
y0 = None
y1 = None
yt = None
px0 = None
py0 = None
loss = None
grad = None
g_last = None
g = None
xstar = None
pxstar = None
sgrad = None
print("Success rate =",suc_tot/200)

    
    