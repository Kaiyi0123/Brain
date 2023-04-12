!apt-get install -y -qq software-properties-common python-software-properties module-init-tools
!add-apt-repository -y ppa:alessandro-strada/ppa 2>&1 > /dev/null
!apt-get update -qq 2>&1 > /dev/null
!apt-get -y install -qq google-drive-ocamlfuse fuse
from google.colab import auth
auth.authenticate_user()
from oauth2client.client import GoogleCredentials
creds = GoogleCredentials.get_application_default()
import getpass
!google-drive-ocamlfuse -headless -id={creds.client_id} -secret={creds.client_secret} < /dev/null 2>&1 | grep URL
vcode = getpass.getpass()
!echo {vcode} | google-drive-ocamlfuse -headless -id={creds.client_id} -secret={creds.client_secret}


!mkdir -p drive
!google-drive-ocamlfuse drive 


import os

path = "/content/drive/XAI/" #filepath
os.chdir(path)
os.listdir(path)


filepath='/content/drive/XAI/data/baseset/'
saved_filepath='XAI'

if not os.path.exists(saved_filepath+'output/'):
    os.makedirs(saved_filepath+'output/')

if not os.path.exists(saved_filepath+'output/model/'):
    os.makedirs(saved_filepath+'output/model/')

if not os.path.exists(saved_filepath+'output/loss/'):
    os.makedirs(saved_filepath+'output/loss/')



import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import time
import scipy.misc
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import torchvision
from torchvision import transforms,models


transf=transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

class MyDataset(Dataset): 
    def __init__(self): 

        df=pd.read_csv(filepath+'info.csv',encoding="gbk")
        self.image_list = df['ImageDataID'].drop_duplicates().reset_index().rename(columns = {'index':'col',0:'ImageDataID'})
        self.demo = df.iloc[self.image_list['col']][['Sex','Age']].reset_index(drop = True)
        self.demo = self.demo.replace('M',0) 
        self.demo = self.demo.replace('F',1)

        self.labels = np.array(self.demo['Age'])
 
    def __getitem__(self, index):   
        id_image = self.image_list['ImageDataID'][index]
        
        label = self.labels[index]


        img=cv2.imread(filepath+id_image+'.png')
        size=(196,160)
        img=cv2.resize(img,size)
        img=transf(img)
        img=img.to(device)
        return torch.tensor(label), img
        
 
    def __len__(self): 
        return len(self.labels)


dataset=MyDataset()

test_num = int(len(dataset)/7)
train_num = len(dataset) - test_num
training_set, test_set = torch.utils.data.random_split(dataset,[train_num, test_num])

train_loader = torch.utils.data.DataLoader(training_set, batch_size=5, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=5, shuffle=True)

loss_function = nn.L1Loss(reduction = 'sum')
train_loss_list = []
test_loss_list = []


model = models.resnet18(pretrained=True)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 1) 
device=torch.device("cuda")
model = model.to(device)

criterion = nn.L1Loss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)


num_epochs = 3 #change epoch numbers
start_time = time.time()

for epoch in range(num_epochs):
    """ Training  """
    model.train()

    running_loss = 0.
    running_corrects = 0

    # load a batch data of images
    for i, (inputs, labels) in enumerate(train_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)

        # get loss value and update the network weights
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / len(training_set)
    #epoch_acc = running_corrects / len(training_set) * 100.
    print('[Train #{}] Acc: {:.4f} Loss: {:.4f} Time: {:.4f}s'.format(epoch, epoch_loss, time.time() - start_time))

    """ Validation"""
    model.eval()

    with torch.no_grad():
        running_loss = 0.
        running_corrects = 0

        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(test_set)
        #epoch_acc = running_corrects / len(test_set) * 100.
        print('[Validation #{}] Acc: {:.4f} Loss: {:.4f} Time: {:.4f}s'.format(epoch, epoch_loss, time.time() - start_time))
