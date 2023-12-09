import zipfile
import os

import torchvision.transforms as transforms

# once the images are loaded, how do we pre-process them before being passed into the network
# by default, we resize the images to 64 x 64 in size
# and normalize them to mean = 0 and standard-deviation = 1 based on statistics collected from
# the training set

#%%validation dataset operation
import shutil
path_data = "/content/drive/MyDrive/Birds/bird_dataset"

for entry in os.listdir(path_data+"/train_images"):
  curr_class_train = path_data+"/train_images/"+entry
  n_train = len(os.listdir(curr_class_train))
  curr_class_val = path_data+"/val_images/"+entry
  n_val = len(os.listdir(curr_class_val))
  int_percentage = int ((n_train+n_val)*0.15-n_val)
  if (int_percentage>0):
    for img in os.listdir(curr_class_train):
      if int_percentage>0:
        shutil.move(curr_class_train+"/"+img, curr_class_val+"/"+img) # moving img from train to val
        int_percentage = int_percentage - 1
      else:
        break
#%%
train_transforms = transforms.Compose([
    transforms.Resize((400, 400)),
    transforms.RandomRotation(degrees= 30 ),
    transforms.RandomHorizontalFlip(p=0.4),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
])

val_test_transforms = transforms.Compose([
    transforms.Resize((400, 400)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
])



