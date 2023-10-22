from torchvision import datasets
import torchvision.transforms as transforms
myTransforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomResizedCrop((224,224),scale=(0.8,1.0),ratio=(1.0,1.0)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
testTransforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
cifar_train = datasets.CIFAR10('cifar', True,  transform=myTransforms,  download=True)
cifar_test = datasets.CIFAR10('cifar', False, transform=testTransforms, download=True)