import os
import argparse
import logging
import torch
import torch.nn as nn
from model.attention_model import ARSC_NET
from model.ResNet_att import model_att
from model.ResNet_att_scratch import model_att_scratch
from model.ResNet_scratch import resnet_scratch
from model.ResNet import model
from data import cifar_train,cifar_test
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser()
parser.add_argument('--nepochs', type=int, default=50)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--save', type=str, default='log/')
parser.add_argument('--debug', action='store_true')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--model', type=str, default="resnet_scratch")
args = parser.parse_args()

def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)

def get_logger(logpath, displaying=True, saving=True, debug=False):
    logger = logging.getLogger()
    if debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logger.setLevel(level)
    if saving:
        info_file_handler = logging.FileHandler(logpath, mode="a")
        info_file_handler.setLevel(level)
        logger.addHandler(info_file_handler)
    if displaying:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        logger.addHandler(console_handler)

    return logger


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    
    makedirs(args.save)
    logger = get_logger(logpath=os.path.join(args.save, 'logs.txt'))
    logger.info(args)

    device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
    if args.model == "resnet50":
        model = model.to(device)
    elif args.model == "ARSC_NET":
        model = ARSC_NET.to(device)
    elif args.model=="resnet_att":
        model = model_att.to(device)
    elif args.model=="resnet_att_scratch":
        model = model_att_scratch.to(device)
    elif args.model=="resnet_scratch":
        model = resnet_scratch.to(device)
    else:
        print("没有设置该模型")
        exit()

    train_loader = DataLoader(dataset=cifar_train, batch_size=args.batch_size, shuffle=True, drop_last=True)   # 加载数据集
    test_loader = DataLoader(dataset=cifar_test, batch_size=args.batch_size, shuffle=True, drop_last=True)   # 加载数据集

    criterion = nn.CrossEntropyLoss().to(device)

    optimizer = torch.optim.SGD(model.parameters(),lr=args.lr, momentum=0.9)
    
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    
    epoch=0
    best_accuracy = 0
    logger.info(model)
    logger.info('Number of parameters: {}'.format(count_parameters(model)))
    model.train()
    for epoch in range(args.nepochs):
        total_loss = 0
        right_number = 0
        logger.info(f"epoch:{format(epoch+1)}")
        for (data,label) in train_loader:
            optimizer.zero_grad()
            data = data.to(device)
            y=model(data).float()
            label = torch.nn.functional.one_hot(label, num_classes=10).float().to(device)
            loss = criterion(y, label)
            lossdata=loss.cpu().detach().numpy()
            total_loss += lossdata
            loss.backward()
            optimizer.step()

            predicted = torch.max(y.data,1)[1]
            labeled = torch.max(label.data,1)[1]
            right_number += (predicted == labeled).sum()
        
        scheduler.step()
        train_acc=right_number/len(train_loader.dataset)
        epoch+=1
        train_loss = total_loss  / (len(train_loader.dataset)/args.batch_size)
        logger.info(f"Train loss : {format(train_loss, '.4f')}\tTrain Acc : {format(train_acc, '.4f')}")
        
        if epoch % 5 == 0:
            model.eval()
            with torch.no_grad():
                test_total_loss=0
                test_right=0
                for (data,label) in test_loader:
                    data = data.to(device)
                    y=model(data).float()
                    label = torch.nn.functional.one_hot(label, num_classes=10).float().to(device)
                    loss = criterion(y, label)
                    lossdata=loss.cpu().detach().numpy()
                    test_total_loss += lossdata
                    predicted = torch.max(y.data,1)[1]
                    labeled = torch.max(label.data,1)[1]
                    test_right += (predicted == labeled).sum()

                test_acc=test_right/len(test_loader.dataset)
                test_loss = test_total_loss  / (len(test_loader.dataset)/args.batch_size)
                logger.info(f"Val loss : {format(test_loss, '.4f')}\tVal Acc : {format(test_acc, '.4f')}") 
                if test_acc>best_accuracy:
                    best_accuracy=test_acc
                    torch.save(model.state_dict(),"ResNet_scratch.pth")

            logger.info(f"best accuracy is {format(best_accuracy, '.4f')}")
