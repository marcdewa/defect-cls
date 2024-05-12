import timm
import torch
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import Dataset
import numpy as np
import cv2

from torchsummary import summary

model_path = "/vcl1/marc/cls-checkpoints/resnet50.tv_in1k_strat1/v5-2024-04-16 11:54:20.238175+09:00_mixup/ep49.ckpt"

model_name = "resnet50.tv_in1k"

model = timm.create_model(model_name, pretrained=False,num_classes=1).to("cuda")
print(model.default_cfg["mean"])
(summary(model))
input_size=(model.default_cfg["input_size"][1],model.default_cfg["input_size"][2])

#load ckpt
checkpoint = torch.load(model_path)
model.load_state_dict(checkpoint)

root_dir = "/vcl1/marc"
version = "v2"

testdir_defects = "{}/defects-classification/BF/datas{}/test_defects".format(root_dir,version)
testdir_goods = "{}/defects-classification/BF/datas{}/test_goods".format(root_dir,version)

test_transforms = transforms.Compose([transforms.Resize(input_size),
                                      transforms.ToTensor(),
                                      torchvision.transforms.Normalize(
                                          mean=model.default_cfg["mean"],
                                          std=model.default_cfg["std"],
    ),
                                      ])



#datasets
test_data_defects = datasets.ImageFolder(testdir_defects,transform=test_transforms)
test_data_goods = datasets.ImageFolder(testdir_goods,transform=test_transforms)

#dataloader
test_data_defects = torch.utils.data.DataLoader(test_data_defects, shuffle = True, batch_size=16)
test_data_goods = torch.utils.data.DataLoader(test_data_goods, shuffle = True, batch_size=16)

def test_step(test_dl,goods=False):
  with torch.no_grad():
    cum_loss = 0
    test_correct=0
    for x_batch, y_batch in test_dl:
      if goods:
          for i,x in enumerate(y_batch):
              y_batch[i]=1
      x_batch = x_batch.to("cuda")
      y_batch = y_batch.unsqueeze(1).float() #convert target to same nn output shape
      y_batch = y_batch.to("cuda")

      #model to eval mode
      model.eval()

      yhat = model(x_batch)

      outs = torch.sigmoid(yhat)
      outs = (outs>0.5).float()
      test_correct += (outs == y_batch).float().sum()
    return 100 * test_correct / (len(test_dl)*16)

print("Defects acc: {}".format(test_step(test_data_defects).item()))
print("Goods acc: {}".format(test_step(test_data_goods,goods=True).item()))

