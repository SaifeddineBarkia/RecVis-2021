# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 18:56:15 2021

@author: Saif
"""

import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets,models
from torch.autograd import Variable
from tqdm import tqdm
import torchvision
import os
import PIL.Image as Image
import torchvision.transforms as transforms
import torch
import numpy as np
#%%
coco_names = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

data_path = "bird_dataset"
FasteRCNN = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')
FasteRCNN.eval().to(device)

Tensor = transforms.Compose([    
    transforms.ToTensor(),
])

def boxed_bird(image, model, label=16,threshold = 0.5):
    image = Tensor(image).to(device)
    image = image.unsqueeze(0) 
    outputs = model(image) 
    index = -1
    labels = outputs[0]['labels'].detach().cpu().numpy()
    scores = outputs[0]['scores'].detach().cpu().numpy() 
    boxes =  outputs[0]['boxes'].detach().cpu().numpy() 
    for i in range(len(labels)):
      if labels[i] == 16:
        if scores[i]>=threshold:
          threshold = scores[i] # keep the best threshold if more than 1 brid is detected
          index = i

    if index != -1: 
      return boxes[index]
    else :
      return np.zeros(4,)


#%% Overwrite our dataset
data_path = "bird_dataset"
a=np.zeros(4,)
# train
for classes in os.listdir(data_path+"/train_images"):
    path_class= data_path+"/train_images/" + classes
    for path_image in os.listdir(path_class):
        image = Image.open(path_class+"/"+path_image)
        shape_img = image.size
        box = boxed_bird(image,FasteRCNN)
        if( not np.allclose(box,a)) :
            box =[max(int(box[0])-10,0), max(int(box[1])-10,0), min(int(box[2])+10,shape_img[0]), min(int(box[3])+10, shape_img[1])] # to take into account detection error.
            crop_img = image.crop(box)
            crop_img.save(path_class+"/"+path_image)
#%%val
for classes in os.listdir(data_path+"/val_images"):
    path_class= data_path+"/val_images/" + classes
    for path_image in os.listdir(path_class):
        image = Image.open(path_class+"/"+path_image)
        shape_img = image.size
        box = boxed_bird(image,FasteRCNN)
        if( not np.allclose(box,a)) :
            box =[max(int(box[0])-10,0), max(int(box[1])-10,0), min(int(box[2])+10,shape_img[0]), min(int(box[3])+10, shape_img[1])] # to take into account detection error.
            crop_img = image.crop(box)
            crop_img.save(path_class+"/"+path_image)
#%%test
path_test = "bird_dataset/test_images/mistery_category"
for path_image in os.listdir(path_test):
        image = Image.open(path_test+"/"+path_image)
        shape_img = image.size
        box = boxed_bird(image,FasteRCNN)
        if( not np.allclose(box,a)) :
            box =[max(int(box[0])-10,0), max(int(box[1])-10,0), min(int(box[2])+10,shape_img[0]), min(int(box[3])+10, shape_img[1])] # to take into account detection error.
            crop_img = image.crop(box)
            crop_img.save(path_test+"/"+path_image)
        else: 
            print(path_image)