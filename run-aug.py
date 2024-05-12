import timm
from tqdm import tqdm
import torch
import torchvision
from torchvision import datasets, transforms
from torchvision.transforms import v2
import numpy as np
import cv2
import csv

import torch.nn.functional as F


from torch.nn.modules.loss import BCEWithLogitsLoss, CrossEntropyLoss
from torchsummary import summary
import os
from itertools import zip_longest

import datetime
import pytz

#Things to change
# model_name = "mobilenetv3_small_100.lamb_in1k"
# model_name = "efficientnet_b1.ft_in1k"
# model_name = "vit_base_patch16_224_miil.in21k"
# model_name = "vit_base_patch16_224_miil.in21k"
# model_name = "vit_large_patch32_384"
model_name = "resnet101.tv_in1k"
#model_name = "resnet50.tv_in1k"
# model_name = "tf_efficientnetv2_l.in21k"
# model_name = "vit_large_patch32_384.orig_in21k_ft_in1k"
# model_name = "swinv2_base_window12_192.ms_in22k"


lr = 0.005
n_epochs = 55

strat = "5"
version = "v2" #v2 or empty v1

root_dir = "/vcl1/marc"

traindir = "{}/defects-classification/BF/datas{}/strat_{}/train".format(root_dir,version,strat)
testdir = "{}/defects-classification/BF/datas{}/strat_{}/test".format(root_dir,version,strat)

# init
model = timm.create_model(model_name, pretrained=True,num_classes=2)
(summary(model))
if "resnet" in model_name:
  input_size = (256,256)
else:
  input_size=(model.default_cfg["input_size"][1],model.default_cfg["input_size"][2])


train_transforms = v2.Compose([v2.Resize(input_size),
                                       v2.ToTensor(),                                
                                       v2.Normalize(
                                           mean=model.default_cfg["mean"],
                                           std=model.default_cfg["std"],
    ),
                                       ])

mixup = v2.MixUp(num_classes=2)
cutmix = v2.CutMix(num_classes=2)

test_transforms = transforms.Compose([v2.Resize(input_size),
                                      v2.ToTensor(),
                                      v2.Normalize(
                                          mean=model.default_cfg["mean"],
                                          std=model.default_cfg["std"],
    ),
                                      ])



#datasets
train_data = datasets.ImageFolder(traindir,transform=train_transforms)
test_data = datasets.ImageFolder(testdir,transform=test_transforms)
print(train_data.class_to_idx)
print(test_data.class_to_idx)

#dataloader
trainloader = torch.utils.data.DataLoader(train_data, shuffle = True, batch_size=16)
testloader = torch.utils.data.DataLoader(test_data, shuffle = True, batch_size=16)

def make_train_step(model, optimizer, loss_fn):
  def train_step(x,y):
    #make prediction
    yhat = model(x)
    #enter train mode
    model.train()
    #compute loss
    loss = loss_fn(yhat,y)

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    #optimizer.cleargrads()

    return loss, yhat
  return train_step


device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
model = model.to(device)
for params in model.parameters():
  params.requires_grad_ = False



#loss
loss_fn = CrossEntropyLoss() #binary cross entropy with sigmoid, so no need to use sigmoid in the model
#optimizer
optimizer = torch.optim.AdamW(model.parameters(),lr=lr)

#scheduler
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[35], gamma=0.1)

#train step
train_step = make_train_step(model, optimizer, loss_fn)



losses = []
val_losses = []

epoch_train_losses = []
epoch_test_losses = []

best_epoch = 0
best_test_accuracy = 0
loggings = {
    "model_name":["model_name"],
    "strat": ["strat"],
    "lr": ["lr"],
    "epoch": ["epoch"],
    "train_loss": ["train_loss"],
    "val_loss": ["val_loss"],
    "train_acc": ["train_acc"],
    "test_acc": ["test_acc"],
    "best_epoch": ["best_epoch"],
    "best_eval_acc": ["best_eval_acc"],
}

def get_ver(dir):
    return str(len(os.listdir(dir))+1)

save_path1 = os.path.join("..","cls-checkpoints",model_name+"_strat"+strat)
os.makedirs(save_path1 ,exist_ok=True)
save_path = os.path.join(
    save_path1,"v"+get_ver(save_path1)+"-"+str(datetime.datetime.now(pytz.timezone('Asia/Seoul')))
        )


print(f"Learning rate = {lr}")
for epoch in range(n_epochs):
  epoch_loss = 0
  train_correct=0
  for i ,data in tqdm(enumerate(trainloader), total = len(trainloader)): #iterate over batches
    x_batch , y_batch_bfr = data
   
    x_batch , y_batch = mixup(x_batch , y_batch_bfr)
    x_batch = x_batch.to(device) #move to gpu
    # y_batch = y_batch.unsqueeze(1).float() #convert target to same nn output shape
    y_batch = y_batch.to(device).float() #move to gpu
    y_batch_bfr =  y_batch_bfr.to(device).float()


    loss, outs = train_step(x_batch, y_batch)
    epoch_loss += loss/len(trainloader)
    losses.append(loss)
    outs = torch.argmax(outs,dim=1)
    train_correct += (outs == y_batch_bfr).float().sum()
  train_accuracy = 100 * train_correct / (len(trainloader)*16)
    
  epoch_train_losses.append(epoch_loss)

  #validation doesnt requires gradient
  with torch.no_grad():
    cum_loss = 0
    test_correct=0
    for x_batch, y_batch in testloader:
      x_batch = x_batch.to(device)
      # y_batch = y_batch.unsqueeze(1).float() #convert target to same nn output shape
      y_batch = y_batch.to(device)

      #model to eval mode
      model.eval()

      yhat = model(x_batch)
      val_loss = loss_fn(yhat,y_batch)
      cum_loss += loss/len(testloader)
      val_losses.append(val_loss.item())

      outs = torch.argmax(yhat,dim=1)
      test_correct += (outs == y_batch).float().sum()
    test_accuracy = 100 * test_correct / (len(testloader)*16)

    if test_accuracy>best_test_accuracy:
      best_epoch = epoch
      best_test_accuracy = test_accuracy

    epoch_test_losses.append(cum_loss)
    print('Epoch : {},train loss : {} , val loss : {}, train acc : {}, eval acc : {}, best eval acc : {}'.format(epoch+1,epoch_loss,cum_loss,train_accuracy,test_accuracy,best_test_accuracy)) 
    #save best model

    #Logging
    loggings['model_name'].append(model_name)
    loggings['strat'].append(strat)
    loggings['lr'].append(lr)
    loggings['epoch'].append(epoch+1)
    loggings['train_loss'].append(epoch_loss.item())
    loggings['val_loss'].append(cum_loss.item())
    loggings['train_acc'].append(train_accuracy.item())
    loggings['test_acc'].append(test_accuracy.item())
    loggings['best_epoch'].append(best_epoch)
    loggings['best_eval_acc'].append(best_test_accuracy)
    
    # best_loss = min(epoch_test_losses)
    

    
    # #early stopping
    # early_stopping_counter = 0
    # if cum_loss > best_loss:
    #   early_stopping_counter +=1

    # if (early_stopping_counter == early_stopping_tolerance) or (best_loss <= early_stopping_threshold):
    #   print("/nTerminating: early stopping")
    #   break #terminate training
  scheduler.step()
  print(scheduler.get_last_lr())
  os.makedirs(save_path ,exist_ok=True)
  torch.save(model.state_dict(),os.path.join(save_path,"ep"+str(epoch)+".ckpt" ))

with open(os.path.join(save_path,"logs.csv"), 'w',newline='') as csvfile:
    csvwriter = csv.writer(csvfile, delimiter='|')
    datas = []
    for k in loggings.keys():
        datas.append(loggings[k])
    exported_datas = zip_longest(*datas, fillvalue = '')
    csvwriter.writerows(exported_datas)

print(best_epoch)
