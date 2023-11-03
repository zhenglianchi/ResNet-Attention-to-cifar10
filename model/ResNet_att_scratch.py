from torchvision import models
from torch import nn
from attention_model import Residual_attention
model_att_scratch = models.resnet50(pretrained=True) #pretrained=True 加载模型以及训练过的参数

model_att_scratch.layer4 = nn.Sequential(model_att_scratch.layer4,
                                 nn.Conv2d(in_channels=2048,out_channels=256,kernel_size=3,padding=1),
                                 Residual_attention(in_planes=256,out_planes=64))
#保持in_features不变，修改out_features=10
model_att_scratch.fc = nn.Sequential(nn.Linear(64,10,bias=True),
                            nn.LogSoftmax(dim=1))