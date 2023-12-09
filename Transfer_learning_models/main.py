import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets,models
from torch.autograd import Variable
from tqdm import tqdm
from efficientnet_pytorch import EfficientNet

use_cuda = torch.cuda.is_available()

# Data initialization and loading
from data import train_transforms,val_test_transforms

batch_size = 16

data_path = "bird_dataset"
train_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(data_path+'/train_images',
                         transform=train_transforms),
    batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(data_path + '/val_images',
                         transform=val_test_transforms),
    batch_size=batch_size, shuffle=False)

# Neural network and optimizer
# We define neural net in model.py so that it can be reused by the evaluate.py script
from model import MyResnet,set_parameter_requires_grad
#%% Resnet50  model
pretrained_resnet = models.resnet50(pretrained=True)
set_parameter_requires_grad(pretrained_resnet)
model = MyResnet(my_pretrained_model = pretrained_resnet )
#%% SeNet50 model
'''import torch.hub
pretrained_resnet = torch.hub.load(
    'moskomule/senet.pytorch',
    'se_resnet50',
    pretrained=True,)
set_parameter_requires_grad(pretrained_resnet)
model = MyResnet(my_pretrained_model = pretrained_resnet )'''

#%% Resnet101 model
'''pretrained_resnet101 = models.resnet101(pretrained=True)
set_parameter_requires_grad(pretrained_resnet101)
model = MyResnet(my_pretrained_model = pretrained_resnet101 )'''

#%%Efficientnet b5 model
'''model = EfficientNet.from_pretrained('efficientnet-b5', num_classes=20)'''

#%%
if use_cuda:
    print('Using GPU')
    model.cuda()
else:
    print('Using CPU')

lr = 0.005
momentum=0.8
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
log_interval = batch_size

def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data)
        criterion = torch.nn.CrossEntropyLoss(reduction='mean')
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval/2 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data.item()))

def validation():
    model.eval()
    validation_loss = 0
    correct = 0
    for data, target in val_loader:
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        output = model(data)
        # sum up batch loss
        criterion = torch.nn.CrossEntropyLoss(reduction='mean')
        validation_loss += criterion(output, target).data.item()
        # get the index of the max log-probability
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
    validation_acc = 100. * correct / len(val_loader.dataset)
    validation_loss /= len(val_loader.dataset)
    print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        validation_loss, correct, len(val_loader.dataset),
        validation_acc))
    return validation_acc

epochs = 100
folder = "model_path"
accuracies = []
best_acc = 0
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max',
                                                 factor=5, 
                                                 patience = 0.8,
                                                 verbose = False)
for epoch in range(1, epochs + 1):
      train(epoch)
      acc = validation()
      accuracies.append(acc)
      scheduler.step(acc)
      model_file = folder+'Resnet101_Cropped_data_best_model'+'.pth'
      if acc > best_acc:
        torch.save(model.state_dict(), model_file)
        best_acc = acc
        early_stopping_after = 20
      else:
        early_stopping_after -= 1 
      
      if early_stopping_after==0:
        torch.save(model.state_dict(), "Resnet101_Cropped_data_Last_checkpoint")
        print("Early Stopping at epoch", epoch)
        break
