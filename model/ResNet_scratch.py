from torchvision import models
from torch import nn
resnet_scratch = models.resnet50(pretrained=True) #pretrained=True 加载模型以及训练过的参数
num_ftrs = resnet_scratch.fc.in_features 
#保持in_features不变，修改out_features=10
resnet_scratch.fc = nn.Sequential(nn.Linear(num_ftrs,10,bias=True),
                            nn.LogSoftmax(dim=1))