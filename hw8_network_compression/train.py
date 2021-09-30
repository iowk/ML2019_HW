import numpy as np
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
      
x = np.load('data/x_train.npy').astype('uint8')
y = np.load('data/y_train.npy').astype(int)
train_x = []
val_x = []
for i in range(25839):    
    img = Image.fromarray(x[i][0],'L')
    train_x.append(img)
for i in range(25839,28709):
    img = Image.fromarray(x[i][0],'L')
    val_x.append(img)
train_y = y[0:25839]
val_y = y[25839:28709]
      
batch_size = 512
epochs = 500

model = Net().half()
#model.load_state_dict(torch.load('model_q3-3.pkl'))
pp=0
for p in list(model.parameters()):
    n=1
    for s in list(p.size()):
        n = n*s
    pp += n
print(pp)
optimizer = optim.Adam(model.parameters(),lr=0.001,eps=1e-04)
aug = transforms.Compose([                          
                          transforms.RandomAffine(degrees=30, scale=(0.8,1.2), shear=30),                                       
                          transforms.RandomHorizontalFlip(),                                                       
                          transforms.ToTensor(),
                          ])   
val_aug = transforms.Compose([                                                                                                    
                          transforms.ToTensor(),
                          ])
train_dataset = MyDataset(train_x,train_y,aug,True) 
train_loader = Data.DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True)
val_dataset = MyDataset(val_x,val_y,val_aug,False)
val_loader = Data.DataLoader(dataset=val_dataset,batch_size=len(val_dataset),shuffle=False)
criterion = nn.CrossEntropyLoss().cuda()
maxi = 0
for epoch in range(epochs):
    print("Epoch:",epoch) 
    tot_loss = 0
    for step,data in enumerate(train_loader,0): 
        batch_x,batch_y = data                        
        batch_x = Variable(batch_x).cuda().half()
        batch_y = Variable(batch_y).cuda()
        model.train().cuda()
        model.zero_grad()
        optimizer.zero_grad()
        output = model(batch_x)
        #print(batch_y.shape)                
        loss = criterion(output,batch_y)        
        loss.backward()
        optimizer.step()
        tot_loss+=loss.item()
        print("step:",step,"loss:",loss.item(),end='\r')
    #validation
    model.eval().cuda()    
    cor = 0
    tot = 0                           
    with torch.no_grad():
        for batch_x,batch_y in val_loader:                                 
            output = model(batch_x.cuda().half())                
            output = output.cpu().numpy()            
            for i in range(output.shape[0]):                        
                pred = np.argmax(output[i])
                real = batch_y[i]
                #if tot==0: print(output,real)
                tot+=1
                if(pred==real): cor+=1 
    val_acc = float(cor/tot)
    print("loss:",tot_loss/25839)              
    print("val acc:",val_acc)    
    if val_acc > maxi:
        maxi = val_acc            
        torch.save(model.state_dict(),'model_123.pkl')     
print("Training complete.")
print("best val acc =",maxi)

x = None
y = None
train_x = None
train_y = None
val_x = None
val_y = None
