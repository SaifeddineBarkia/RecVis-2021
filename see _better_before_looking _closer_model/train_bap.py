
import os
import time
import shutil
import random
import numpy as np
from tqdm import tqdm

# include from folder
from model.inception_bap import inception_v3_bap
from model.resnet import resnet50
from dataset.custom_dataset import CustomDataset

from utils import calculate_pooling_center_loss, mask2bbox
from utils import attention_crop, attention_drop, attention_crop_drop
from utils import getDatasetConfig, getConfig
from utils import accuracy, get_lr, save_checkpoint, AverageMeter, set_seed
from utils import Engine
from prettytable import PrettyTable
# include from pytorch 
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision.models as models
import torch.nn.functional as F
from torchvision import datasets
from PIL import Image

GLOBAL_SEED = 1231

def image_loader(loader, image_name):
    image = Image.open(image_name).convert('RGB')
    image = loader(image).float()
    image = torch.tensor(image, requires_grad=True)
    image = image.unsqueeze(0)
    return image

def count_parameters(model):
        table = PrettyTable(["Modules", "Parameters"])
        total_params = 0
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad: continue
            param = parameter.numel()
            table.add_row([name, param])
            total_params+=param
        print(table)
        print(f"Total Trainable Params: {total_params}")
        return total_params

def set_parameter_requires_grad(model, feature_extracting=True):
    if feature_extracting:
        model_ft = models.resnet50(pretrained=True)
        ct = 0
        for name, child in model.named_children():
          ct += 1
          if ct < 6:
              for name2, params in child.named_parameters():
                params.requires_grad = False

def train():
    # input params
    set_seed(GLOBAL_SEED)
    config = getConfig()
    best_prec1 = 0.
    data_path = "bird_dataset"
    # define train_dataset and loader
    
    data_path = "/content/drive/MyDrive/Birds/bird_dataset"
    
    transform_train = transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.RandomRotation(degrees= 30 ),
        transforms.RandomHorizontalFlip(p=0.4),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    train_dataset = datasets.ImageFolder(data_path+'/train_images',transform=transform_train)
    train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=config.batch_size, shuffle=True )
    
    transform_val = transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_dataset = datasets.ImageFolder(data_path+'/val_images',transform=transform_val)
    
    val_loader = torch.utils.data.DataLoader(val_dataset,batch_size=config.batch_size, shuffle=False )
    
    # logging dataset info
    print('Dataset Name:{dataset_name}, Train:[{train_num}], Val:[{val_num}]'.format(
        dataset_name=config.dataset,
        train_num=len(train_dataset),
        val_num=len(val_dataset)))
    print('Batch Size:[{0}], Total:::Train Batches:[{1}],Val Batches:[{2}]'.format(
        config.batch_size, len(train_loader), len(val_loader)
    ))
    
    # define model
    if config.model_name == 'inception':
        net = inception_v3_bap(pretrained=True, aux_logits=False,num_parts=config.parts)
    elif config.model_name == 'resnet50':
        net = resnet50(pretrained=True,use_bap=True)

    set_parameter_requires_grad(net)
    in_features = net.fc_new.in_features
    new_linear = torch.nn.Linear(
        in_features=in_features, out_features=20)
    net.fc_new = new_linear
    
    # feature center
    feature_len = 768 if config.model_name == 'inception' else 512
    center_dict = {'center': torch.zeros(
        20, feature_len*config.parts)}
    print(count_parameters(net))

    # gpu config
    use_gpu = torch.cuda.is_available() and config.use_gpu
    if use_gpu:
        net = net.cuda()
        center_dict['center'] = center_dict['center'].cuda()
    gpu_ids = [int(r) for r in config.gpu_ids.split(',')]
    if use_gpu and config.multi_gpu:
        net = torch.nn.DataParallel(net, device_ids=gpu_ids)

    # define optimizer
    assert config.optim in ['sgd', 'adam'], 'optim name not found!'
    if config.optim == 'sgd':
        optimizer = torch.optim.SGD(
            net.parameters(), lr=config.lr, momentum=config.momentum, weight_decay=config.weight_decay)
    elif config.optim == 'adam':
        optimizer = torch.optim.Adam(
            net.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    # define learning scheduler
    
    assert config.scheduler in ['plateau',
                                'step'], 'scheduler not supported!!!'
    if config.scheduler == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 'min', patience=3, factor=0.1)
    elif config.scheduler == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=2, gamma=0.9)

    # define loss
    criterion = torch.nn.CrossEntropyLoss()
    if use_gpu:
        criterion = criterion.cuda()

    # train val parameters dict
    net.load_state_dict(torch.load(config.checkpoint_path+'/checkpoint.pth.tar')['state_dict'])
    state = {'model': net, 'train_loader': train_loader,
             'val_loader': val_loader, 'criterion': criterion,
             'center': center_dict['center'], 'config': config,
             'optimizer': optimizer}

    ## train and val
    engine = Engine()
    m=0
    print(config)
    for e in range(config.epochs):
        if config.scheduler == 'step':
            scheduler.step()
        lr_val = get_lr(optimizer)
        
        print("Start epoch %d ==========,lr=%f" % (e, lr_val))
        train_prec, train_loss = engine.train(state, e)
        prec1, val_loss = engine.validate(state)
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)

        save_checkpoint({
            'epoch': e + 1,
            'state_dict': net.state_dict(),
            'best_prec1': best_prec1,
            'optimizer': optimizer.state_dict(),
            'center': center_dict['center']
        }, is_best, config.checkpoint_path)
        if config.scheduler == 'plateau':
            scheduler.step(val_loss)

def predict():
    
    engine = Engine()
    config = getConfig()
    data_config = getDatasetConfig(config.dataset)
    transform_test = transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    # define model
    if config.model_name == 'inception':
        net = inception_v3_bap(pretrained=True, aux_logits=False)
    elif config.model_name == 'resnet50':
        net = resnet50(pretrained=True)

    in_features = net.fc_new.in_features
    new_linear = torch.nn.Linear(
        in_features=in_features, out_features=20)
    net.fc_new = new_linear

    # load checkpoint
    use_gpu = torch.cuda.is_available() and config.use_gpu
    gpu_ids = [int(r) for r in config.gpu_ids.split(',')]
    if use_gpu and len(gpu_ids) > 1:
        net = torch.nn.DataParallel(net, device_ids=gpu_ids)
    net = torch.nn.DataParallel(net)
    net.load_state_dict(torch.load(config.checkpoint_path)['state_dict'])
    net.eval()
    if use_gpu:
        net = net.cuda()
    criterion = torch.nn.CrossEntropyLoss()
    use_cuda = torch.cuda.is_available()
    if use_gpu:
        criterion = criterion.cuda()
    test_dir = '/content/drive/MyDrive/Birds/bird_dataset/test_images/mistery_category'
    def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
        with open(path, 'rb') as f:
            with Image.open(f) as img:
                return img.convert('RGB')
    
    output_file = open("Attention_model_Inceptionv3_cropped.csv", "w")
    output_file.write("Id,Category\n")
    with torch.no_grad():
      for f in tqdm(os.listdir(test_dir)):
          if 'jpg' in f:
              data = transform_test(pil_loader(test_dir + '/' + f))
              data = data.view(1, data.size(0), data.size(1), data.size(2))
              if use_cuda:
                  data = data.cuda()
              output = net(data)
              output = torch.nn.Softmax(dim=1)(output[2])
              pred = output.max(1, keepdim=True)[1]
              output_file.write("%s,%d\n" % (f[:-4], pred))

      output_file.close()
    
    print("Succesfully wrote Kaggle_attentionyou can upload this file to the kaggle competition website")    
if __name__ == '__main__':
    config = getConfig()
    engine = Engine()
    if config.action == 'train':
        train()
    elif config.action == 'test' :
        test()
    else :
      predict()
