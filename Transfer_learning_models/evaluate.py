import argparse
from tqdm import tqdm
import os
import PIL.Image as Image

import torch

from model import MyResnet,set_parameter_requires_grad
from torchvision import datasets, models, transforms
from efficientnet_pytorch import EfficientNet
import torch.hub
from data import val_test_transforms
'''parser = argparse.ArgumentParser(description='RecVis A3 evaluation script')
parser.add_argument('--data', type=str, default='bird_dataset', metavar='D',
                    help="folder where data is located. test_images/ need to be found in the folder")
parser.add_argument('--model', type=str, metavar='M',
                    help="the model file to be evaluated. Usually it is of the form model_X.pth")
parser.add_argument('--outfile', type=str, default='experiment/kaggle.csv', metavar='D',
                    help="name of the output csv file")

args = parser.parse_args()'''
use_cuda = torch.cuda.is_available()

#%% Loading model 1
state_dict = torch.load("model_pathResnet50_Cropped_data_best_model.pth")
pretrained_resnet = models.resnet50(pretrained=True)
model1 = MyResnet(my_pretrained_model = pretrained_resnet )
model1.load_state_dict(state_dict)
model1.eval()
#%% Loading model 2
state_dict = torch.load("model_pathSenet50_Cropped_data_best_model.pth")
pretrained_resnet = torch.hub.load(
    'moskomule/senet.pytorch',
    'se_resnet50',
    pretrained=True,)
model2 = MyResnet(my_pretrained_model = pretrained_resnet )
model2.load_state_dict(state_dict)
model2.eval()
#%% Loading model 3
state_dict = torch.load("model_pathResnet101_Cropped_data_best_model.pth")
pretrained_resnet = models.resnet101(pretrained=True)
model3 = MyResnet(my_pretrained_model = pretrained_resnet )
model3.load_state_dict(state_dict)
model3.eval()
#%% Loading model 4
state_dict = torch.load("experimentsEfficient_Cropped_data_best_model.pth")
model4 = EfficientNet.from_pretrained('efficientnet-b5', num_classes=20)
model4.load_state_dict(state_dict)
model4.eval()
##% sending to device
if use_cuda:
    print('Using GPU')
    model1.cuda()
    model2.cuda()
    model3.cuda()
    model4.cuda()
else:
    print('Using CPU')
#%% evaluating
test_dir = "bird_dataset/test_images/mistery_category"

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


output_file = open("ModelBagging.csv", "w")
output_file.write("Id,Category\n")
for f in tqdm(os.listdir(test_dir)):
    if 'jpg' in f:
        data = val_test_transforms(pil_loader(test_dir + '/' + f))
        data = data.view(1, data.size(0), data.size(1), data.size(2))
        if use_cuda:
            data = data.cuda()
        output1 = model1(data)
        output1 = torch.nn.Softmax(dim=1)(output1)
        output2 = model2(data)
        output2 = torch.nn.Softmax(dim=1)(output2)
        output3 = model3(data)
        output3 = torch.nn.Softmax(dim=1)(output3)
        output4 = model1(data)
        output4 = torch.nn.Softmax(dim=1)(output3)
        output = (output1 + output2 +output3+output4 ) /4
        pred = output.data.max(1, keepdim=True)[1]
        output_file.write("%s,%d\n" % (f[:-4], pred))

output_file.close()

print("Succesfully wrote " + " ModelBagging " + ', you can upload this file to the kaggle competition website')
        


