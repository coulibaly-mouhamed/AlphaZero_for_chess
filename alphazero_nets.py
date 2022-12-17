'''
Implement the AlphaZero algorithm with PyTorch for the Chess game. 
'''
import torch
from torch import nn


'''
Create a convolutional block made of :
    -A convolution of 256 filters of kernel size 3x3 with stride 1 
    - batch normalization
    -A rectifier non linearity 
'''

class ConvBlock(nn.Module):
    
    def __init__(self):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(22, 256, kernel_size=3, stride=1, padding=1)
        self.batch_norm = nn.BatchNorm2d(256)
        
    def forward(self, x):
        x = x.view(-1, 22, 8, 8)
        x  = nn.functional.relu(self.batch_norm(self.conv1(x)))
        
        return x
    
'''
Create a residual block made of :
    -A convolution of 2 filters of kernel size 1*1 with stride 1
    - batch normalization
    - A skip connection that adds the input to the block
    -A rectifier non linearity
'''

class ResBlock(nn.Module):
    
    def __init__(self, inplanes=256, planes=256, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = nn.functional.relu(self.bn1(out))
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = nn.functional.relu(out)
        return out



class OutBlock(nn.Module):
    def __init__(self):
        super(OutBlock, self).__init__()
        self.conv = nn.Conv2d(256, 1, kernel_size=1) # value head
        self.bn = nn.BatchNorm2d(1)
        self.fc1 = nn.Linear(8*8, 64)
        self.fc2 = nn.Linear(64, 1)
        
        self.conv1 = nn.Conv2d(256, 128, kernel_size=1) # policy head
        self.bn1 = nn.BatchNorm2d(128)
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.fc = nn.Linear(8*8*128, 8*8*73)
    
    def forward(self,s):
        v = nn.functional.relu(self.bn(self.conv(s))) # value head
        v = v.view(-1, 8*8)  # batch_size X channel X height X width
        v = nn.functional.relu(self.fc1(v))
        v = nn.functional.tanh(self.fc2(v))
        
        p = nn.functional.relu(self.bn1(self.conv1(s))) # policy head
        p = p.view(-1, 8*8*128)
        p = self.fc(p)
        p = self.logsoftmax(p).exp()
        return p,v

      
        
        
class Alphazero_net(nn.Module):
    '''
    Implement the AlphaZero algorithm with PyTorch for the Chess game. It consists in a 
    Residual Network with 19 layers and 256 filters.
    '''
    def __init__(self,inputs_dim,output_dim):
        super(Alphazero_net, self).__init__()
        self.conv = ConvBlock()
        for block in range(19):
            self.add_module('resblock'+str(block),ResBlock())
        self.outblock = OutBlock()
        
    def forward(self, x):
        s = self.conv(x)
        for block in range(19):
            s = self.__getattr__('resblock'+str(block))(s)
        p,v = self.outblock(s)
        return p,v
    
    def checkpoint(self,epoch):
        torch.save(self.state_dict(), 'checkpoint.pth.tar')
        print('Checkpoint saved !')
        
        
    def train(self,epochs=2000,cpu=1):
        torch.manual_cpu(cpu)
        cuda = torch.cuda.is_available()
        
    
        
        

