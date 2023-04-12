#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#BNU数据一共194张图片，batch设置为?
#训练集3/4 测试集1/4


# In[1]:


get_ipython().system('pip install tqdm')
get_ipython().system('pip install matplotlib')
get_ipython().system('pip install scipy')


# In[2]:


get_ipython().system('pip install nibabel')
get_ipython().system('pip install scikit-image')
get_ipython().system('pip install openpyxl')


# In[1]:


import os 
import sys 
import warnings
import datetime
import logging
import argparse
from tqdm import tqdm

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy.io as sio
import warnings


# In[2]:


import time
import scipy.misc


import nibabel as nib
from skimage import transform


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# In[3]:


torch.cuda.get_device_name(0)


# In[4]:


torch.__version__


# In[5]:


filepath='G:/文件/FUR/DATA/BNU_processed/'
saved_filepath='G:/文件/FUR/DATA/BNU_output'


# In[6]:



if not os.path.exists(saved_filepath+'/model/'):
    os.makedirs(saved_filepath+'/model/')

if not os.path.exists(saved_filepath+'/loss/'):
    os.makedirs(saved_filepath+'/loss/') 


# In[7]:


class argspara():
    def __init__(self):
        self.GPU_num = torch.cuda.device_count()
        self.batch_size = 5
        self.epoch = 60
        self.pretrain = False
        self.lr = 0.0001
        self.split_num = 7.5
        self.pretrain_path = ''


# In[12]:


def conv3x3x3(in_planes, out_planes, stride=1, dilation=1):
    # 3x3x3 convolution with padding
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        dilation=dilation,
        stride=stride,
        padding=dilation,
        bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3x3(inplanes, planes, stride=stride, dilation=dilation)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes, dilation=dilation)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
    
class Res_Model(nn.Module):
    def __init__(self, block, layer_nums):
        super(Res_Model, self).__init__()
        # modality_1
        self.inplanes = 64
        self.conv11 = nn.Conv3d(1,64, kernel_size=3, stride=(2, 2, 2), padding=(1, 1, 1),bias=False)
        self.bn11 = nn.BatchNorm3d(64)
        self.relu11 = nn.ReLU(inplace=True)
        self.maxpool11 = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
        
        self.layer11 = self._make_layer(block, 64, layer_nums[0], flag = 1)
        self.layer12 = self._make_layer(block, 128, layer_nums[1], stride=2)
        self.layer13 = self._make_layer(block, 256, layer_nums[2], stride=1, dilation=2)
        self.layer14 = self._make_layer(block, 512, layer_nums[3], stride=1, dilation=4)
        self.pooling_layer1 = nn.AvgPool3d(kernel_size=(16,16,16), stride=2, padding=0)
        
        # modality_2

        self.layer23 = self._make_layer(block, 256, layer_nums[2], stride=1, dilation=2)
        self.layer24 = self._make_layer(block, 512, layer_nums[3], stride=1, dilation=4)
        self.pooling_layer2 = nn.AvgPool3d(kernel_size=(32,32,32), stride=2, padding=0)
        
        # modality_3

#         self.layer33 = self._make_layer(block, 256, layer_nums[2], stride=1, dilation=2)
#         self.layer34 = self._make_layer(block, 512, layer_nums[3], stride=1, dilation=4)
        #self.pooling_layer3 = nn.AvgPool3d(kernel_size=(32,32,32), stride=2, padding=0)
        
        self.output_layer = nn.Linear(128,1)
        
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1,flag = 0):
        if flag:
            self.inplanes = 64
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False), nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, stride=stride, dilation=dilation, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    
    def forward(self, data):
        #x2,x3
        x1 = data
        
        x1 = self.conv11(x1)
        x1 = self.bn11(x1)
        x1 = self.relu11(x1)
        x1 = self.maxpool11(x1)
        x1 = self.layer11(x1)
        x1 = self.layer12(x1)
        x1 = self.layer13(x1)
        x1 = self.layer14(x1)
#pooling_layer1 kernel 改成了16,16,16
        x1 = self.pooling_layer1(x1)
        
        x1 = torch.flatten(x1, 1)
        
        x2 = self.conv21(x2)
        x2 = self.bn21(x2)
        x2 = self.relu21(x2)
        x2 = self.maxpool21(x2)
        x2 = self.layer21(x2)
        x2 = self.layer22(x2)
        x2 = self.layer23(x2)
        x2 = self.layer24(x2)
        x2 = self.pooling_layer2(x2)
        x2 = torch.flatten(x2, 1)
        
        x3 = self.conv31(x3)
        x3 = self.bn31(x3)
        x3 = self.relu31(x3)
        x3 = self.maxpool31(x3)
        x3 = self.layer31(x3)
        x3 = self.layer32(x3)
        x3 = self.layer33(x3)
        x3 = self.layer34(x3)
        x3 = self.pooling_layer3(x3)
        x3 = torch.flatten(x3, 1)
        
        #x2,x3
        #没有cat
        #final_feature_vector = torch.cat((x1), 1)
        
        res = self.output_layer(x1)
        
        return res


# In[13]:


args = argspara()
para_name = '-'.join([key + ':' + str(value) for key, value in args.__dict__.items()])
if not os.path.exists('logging'):
    os.mkdir('logging')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', filename='logging/'+para_name+'.log')
logger = logging.getLogger(__name__)
logging.info('Start training, parameter:'+para_name)


# In[14]:



if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device('cpu')
model = Res_Model(BasicBlock, [1, 1, 1]).to(device)


# In[21]:


class MyDataset(Dataset): 
    def __init__(self): 
        
        self.filepath='G:/文件/FUR/DATA/IXI_processed/'
        df=pd.read_csv(filepath+'IXI.csv',encoding = "utf-8")
        self.image_list = df['ImageDataID'].drop_duplicates().reset_index().rename(columns = {'index':'col',0:'ImageDataID'})
        self.demo = df.iloc[self.image_list['col']][['Sex','Age']].reset_index(drop = True)
        self.demo = self.demo.replace('M',0) 
        self.demo = self.demo.replace('F',1)

        self.labels = np.array(self.demo['Age'])
        #self.feature = np.load(saved_path+'\resnet10.pth')

    def __getitem__(self, index):   
        id_image = self.image_list['ImageDataID'][index]
        
        label = self.labels[index]

        img_obj = nib.load(filepath+id_image+'.nii.gz').dataobj
        img_data = np.array(img_obj)
        height, width, channels = img_data.shape
        x = height - 128
        x_start = int(x/2)
        x_end = int(128+x/2)
        
        y = width - 128
        y_start = int(y/2) 
        y_end = int(128+y/2)
        
        z = channels -128
        z_start = int(z/2) 
        z_end = int(128+z/2)
        
        img = img_data[:128,:128,z_start:z_end]
        img_tensor = torch.from_numpy(img)
        input_tensor = img_tensor.unsqueeze(0)
        input_tensor = input_tensor.type(torch.FloatTensor)
        
        label_tensor = torch.tensor(label).unsqueeze(0)
        return torch.tensor(label),input_tensor
        
 
    def __len__(self): 
        return len(self.labels)


# In[22]:


optimiser = optim.Adam(model.parameters(), lr=args.lr)
dataset = MyDataset()
test_num = 80
train_num = len(dataset) - test_num
training_set, test_set = torch.utils.data.random_split(dataset, [train_num, test_num])
train_loader = DataLoader(training_set, batch_size=args.batch_size, shuffle=True, pin_memory = True)
test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=True,pin_memory = True)

loss_function = nn.L1Loss(reduction = 'sum')
train_loss_list = []
test_loss_list = []


# In[24]:


total_starttime = datetime.datetime.now()
for epoch in range(args.epoch):
    logging.info('Epoch '+str(epoch) + ' start training')
    running_loss = 0.0
    real_loss = 0.0
    ## 模型训练
    model.train()
    for data in tqdm(train_loader):
        labels,inputs = data
        inputs = inputs.to(device) 
        labels = labels.to(device)
        
        optimiser.zero_grad()

        outputs = model(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimiser.step()

        running_loss += loss.item()
        real_loss += torch.sum(torch.abs(labels - outputs))
        
        
    logging.info("Epoch %d, training loss %4.2f" % (epoch, real_loss/train_num))
    print('Training complete')
    
    train_loss_list.append(float((real_loss/train_num).detach().cpu()))
    ## 模型测试
    model.eval()
    with torch.no_grad():
        test_loss = 0
        predict_list = []
        true_list = []
        for data in tqdm(test_loader):
            labels,inputs = data
            inputs = inputs.to(device) 
            labels = labels.to(device)
            outputs = model(inputs)
            #predict_list.extend(list(outputs.detach().cpu().squeeze().numpy()))
            #true_list.extend(list(labels.detach().cpu().squeeze().numpy()))

            test_loss += torch.sum(torch.abs(labels - outputs))
            
        logging.info('Epoch %d, true:' % epoch + str(true_list) + 'predict:' + str(predict_list))
    if epoch == 0 or (epoch+1) % 5 ==0:
        model_name = saved_filepath+'/model' +'_epoch'+str(epoch)+'.pkl'
        torch.save(model, model_name)
        # np.savez(saved_filepath+'output/loss/Loss' + para_name +'_epoch'+str(epoch)+'.npz',train_loss_list = train_loss_list.data.cpu().numpy(), test_loss_list = test_loss_list.data.cpu().numpy())
    logging.info('Epoch %d Test loss %4.2f' % (epoch, test_loss/test_num))
    test_loss_list.append(float((test_loss/test_num).detach().cpu()))
    
print('**** Finished Training ****')
total_endtime = datetime.datetime.now()
total_elapsed_sec = (total_endtime - total_starttime).total_seconds()
print('\nTotal training time: '+ '{:.2f}'.format(total_elapsed_sec/60) + " minutes.")


# In[25]:


plt.figure(figsize=(20, 10))
plt.plot(train_loss_list, 'r--', label='train')
plt.plot(test_loss_list, 'b--', label='valid')
plt.xlabel("Epoch")
plt.ylabel('Loss')
plt.legend()
plt.title('IXI Dataset Loss of  Resnet10 model with learning rate {}'.format(args.lr))
plt.savefig(saved_filepath+"/loss_history" + "test1" + ".png")

model_name = saved_filepath+'/model' + "test1" + '.pkl'
torch.save(model, model_name)

