import torch
import torch.nn as nn
import torch.nn.functional as F
def conv3x3(in_planes, out_planes, padding=1, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=padding, bias=False)

def conv7x7(in_planes, out_planes, padding=1, stride=1):
    """7x7 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=7, stride=stride, padding=padding, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    """1x convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def norm(dim):
    return nn.BatchNorm2d(num_features=dim)

class ChannelAttention(nn.Module):
    def __init__(self,in_planes=64,ratio=2):
        super(ChannelAttention, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
           
        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes//ratio, 1, bias=False),
                               nn.ReLU(),
                               nn.Conv2d(in_planes//ratio, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        out = self.sigmoid(out)
        return torch.mul(x,out)

class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size=7, stride=1, padding=7//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(out)
        out = self.sigmoid(out)

        return torch.mul(x,out)



class Channel_Spatial_Attention(nn.Module):
    def __init__(self,in_planes):
        super(Channel_Spatial_Attention,self).__init__()

        self.channel=ChannelAttention(in_planes)
        self.Spatial=SpatialAttention()
    
    def forward(self, x):
        x=self.channel(x)
        out=self.Spatial(x)
        return out

class Residual_attention(nn.Module):
    def __init__(self,in_planes=64,out_planes=64,ratio=2):
        super(Residual_attention,self).__init__()

        self.conv1=conv3x3(in_planes,in_planes//ratio,3//2)
        self.conv2=conv3x3(in_planes//ratio,out_planes,3//2)
        self.conv3=conv1x1(in_planes,out_planes)
        self.relu=nn.ReLU()
        self.bn=norm(out_planes)
        self.attention=Channel_Spatial_Attention(out_planes)
    
    def forward(self, x):
        shortcut=self.bn(self.conv3(x))
        out=self.conv1(x)
        out=self.conv2(out)
        out=self.attention(out)
        out=self.relu(out+shortcut)
        return out

class Flatten(nn.Module):

    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        shape = torch.prod(torch.tensor(x.shape[1:])).item()
        return x.view(-1, shape)

class ARSC_NET(nn.Module):
    def __init__(self):
        super(ARSC_NET,self).__init__()
        self.conv1=conv7x7(3,64,7//2)
        self.bn1=norm(64)
        self.relu1=nn.ReLU()
        self.maxpool1=nn.AdaptiveMaxPool2d((64,64))
        self.resattention1=Residual_attention(64,128)
        self.attention=Channel_Spatial_Attention(128)
        self.resattention2=Residual_attention(128,64)
        self.avepool=nn.AdaptiveAvgPool2d((1,1))
        self.flat = Flatten()
        self.fc=nn.Linear(64,10)

    def forward(self, x):
        x=self.conv1(x)
        x=self.bn1(x)
        x=self.relu1(x)
        x=self.maxpool1(x)
        x=self.resattention1(x)

        x=self.attention(x)
        x=self.resattention2(x)
        x=self.avepool(x)
        x=self.flat(x)
        out=F.log_softmax(self.fc(x), dim=1)
        return out