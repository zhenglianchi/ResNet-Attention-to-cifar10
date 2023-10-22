from torchvision import models
from torch import nn
model = models.resnet50(pretrained=True) #pretrained=True 加载模型以及训练过的参数

num_ftrs = model.fc.in_features 
for param in model.parameters():
    param.requires_grad = False #False：冻结模型的参数，也就是采用该模型已经训练好的原始参数。只需要训练我们自己定义的Linear层
 
#保持in_features不变，修改out_features=10
model.fc = nn.Sequential(nn.Linear(num_ftrs,10,bias=True),
                            nn.LogSoftmax(dim=1))