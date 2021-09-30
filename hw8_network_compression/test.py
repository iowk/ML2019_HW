import numpy as np
import random
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
from torch.utils.data.dataset import Dataset
import torchvision.transforms as transforms
from torch.autograd import Variable
from PIL import Image

from model.cnn0602 import Net

class MyDataset(Dataset):
    def __init__(self,imgx,y,transform,istrain):
        self.imgx = imgx
        self.y = y
        self.transform = transform
        self.istrain=istrain
    
    def __getitem__(self,index):        
        x_trans = self.transform(self.imgx[index])        
        return x_trans,self.y[index]
    
    def __len__(self):        
        return len(self.imgx)
      
x = np.load('data/x_test.npy').astype('uint8')
y_ans = np.zeros((7178))
test_x = []
for i in range(7178):    
    img = Image.fromarray(x[i][0],'L')
    test_x.append(img)
      
batch_size = 512

model = Net().half()
model.load_state_dict(torch.load('model_small.pkl', map_location={'cuda:0': 'cpu'}))
model = model.float()  
val_aug = transforms.Compose([                                                                                                    
                          transforms.ToTensor(),                          
                          ])
val_dataset = MyDataset(test_x,y_ans,val_aug,False)
val_loader = Data.DataLoader(dataset=val_dataset,batch_size=len(val_dataset),shuffle=False)

tot = 0
model.eval()
random.seed(1)
with torch.no_grad():
    for batch_x,batch_y in val_loader:                                  
        output = model(batch_x)                
        output = output.numpy()            
        for i in range(output.shape[0]):                        
            pred = np.argmax(output[i])
            y_ans[tot] = pred
            tot+=1

f = open(sys.argv[1],'w')
f.write("id,label"+'\n')
for i in range(y_ans.shape[0]):
    msg = str(i) + "," + str(int(y_ans[i])) + '\n'
    f.write(msg)
    
test_x = None
x = None
y = None

