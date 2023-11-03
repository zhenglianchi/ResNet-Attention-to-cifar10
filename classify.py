from model.ResNet_att import model_att
from PIL import Image
import torchvision.transforms as transforms
import torch

dict={0:"airplane",1:"automobile",2:"bird",3:"cat",4:"deer",5:"dog",
      6:"frog",7:"horse",8:"ship",9:"truck"}
model_att.load_state_dict(torch.load("trained_model/ResNet_att_scratch.pth"))
img_src="image/dog.jpg"
img = Image.open(img_src)
mytransforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
img = mytransforms(img)
img = torch.unsqueeze(img, 0) #给最高位添加一个维度，也就是batchsize的大小
device = torch.device('cuda:' + str(0) if torch.cuda.is_available() else 'cpu')
model_att.to(device)
img = img.to(device)
with torch.no_grad():
    output=model_att(img)
predicted = torch.max(output.data,1)[1]
print(dict[int(predicted[0])])