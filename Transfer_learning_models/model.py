import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, models, transforms

# Function to freeze number of layers
def set_parameter_requires_grad(model, feature_extracting=True,n_layer=6):
    if feature_extracting:
        model_ft = models.resnet50(pretrained=True)
        ct = 0
        for name, child in model.named_children():
          ct += 1
          if ct < n_layer:
              for name2, params in child.named_parameters():
                params.requires_grad = False
    
class MyResnet(nn.Module):
    def __init__(self, my_pretrained_model):
        super(MyResnet, self).__init__()
        self.pretrained = my_pretrained_model
        self.num_ftrs = self.pretrained.fc.in_features
        self.pretrained.fc = nn.Sequential(
        nn.Linear(self.num_ftrs,256),
        nn.ReLU(),
        nn.Dropout(0.05,inplace=True),
        nn.Linear(256,128),
        nn.ReLU(),
        nn.Linear(128,64),
        nn.ReLU(),
        nn.Linear(64,20))
        

    def forward(self, x):
        
        x = self.pretrained(x)
     
        return x
    

#%%
if __name__ == '__main__':
    pretrained_resnet = models.resnet101(pretrained=True)
    set_parameter_requires_grad(pretrained_resnet)
    model = MyResnet(my_pretrained_model = pretrained_resnet )
    print(model)

    from prettytable import PrettyTable
    
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
            
    count_parameters(model)