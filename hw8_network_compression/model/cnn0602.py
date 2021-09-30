import torch
import torch.nn as nn


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()        
        
        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True)
            )

        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),
    
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            )

        self.model = nn.Sequential(
            conv_bn(  1,  32, 2), 
            conv_dw( 32,  64, 1),
            conv_dw( 64, 128, 2),
            conv_dw(128, 128, 1),            
            #conv_dw(128, 128, 1),
            conv_dw(128, 192, 2),
            conv_dw(192, 192, 1),                  
            nn.AvgPool2d(7),
        )
        self.fc = nn.Linear(192, 7)
        self.sm = nn.Softmax(dim=1)

    def forward(self, x): 
        x-=0.5               
        x = self.model(x)
        x = x.view(-1, 192)
        x = self.fc(x)       
        x = self.sm(x)
        #print(x.data.size())       
        return x