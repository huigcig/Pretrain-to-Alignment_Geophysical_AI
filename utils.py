import os,time
import torch
import random
import struct
import numpy as np

import loralib as lora

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.colors import LinearSegmentedColormap
from vispy.color import get_colormap, Colormap, Color
from matplotlib.colors import ListedColormap
import matplotlib.colors as mcolors

from ssim import SSIMLoss as ssim
from ssim import MultiScaleSSIMLoss as mssim
from priorloss import SegmentLoss, NormalLoss

####################### LoRA #######################
class LoRALinear(nn.Module):
    def __init__(self, linear: nn.Linear, r=8, alpha=16):
        super().__init__()
        self.linear = linear
        self.r = r
        self.scaling = alpha / r

        self.lora_A = nn.Parameter(torch.zeros(r, linear.in_features))
        self.lora_B = nn.Parameter(torch.zeros(linear.out_features, r))

        nn.init.kaiming_uniform_(self.lora_A, a=np.sqrt(5))
        nn.init.zeros_(self.lora_B)

        # 冻结原始 Linear
        for p in self.linear.parameters():
            p.requires_grad = False

    def forward(self, x):
        return self.linear(x) + (x @ self.lora_A.T @ self.lora_B.T) * self.scaling

def inject_lora_vit(model, r=8, alpha=16):
    """
    Inject LoRA into ViT Attention (qkv & proj)
    """
    for blk in model.blocks:
        attn = blk.attn
        attn.qkv  = LoRALinear(attn.qkv,  r=r, alpha=alpha)
        attn.proj = LoRALinear(attn.proj, r=r, alpha=alpha)

# def get_lora_decoder_state_dict(model):
#     sd = model.state_dict()
#     return {
#         k: v for k, v in sd.items()
#         if (
#             "lora_" in k
#             or "decoder" in k
#             or "segmentation_head" in k
#         )
#     }
####################### LoRA #######################
        
        
def sort_list_IDs(list_IDs):
    list_nums = [int(i.split(".")[0]) for i in list_IDs]
    list_sort = sorted(enumerate(list_nums), key=lambda x:x[1])
    list_index = [i[0] for i in list_sort]
    list_IDs_new = [list_IDs[i] for i in list_index]
    return list_IDs_new

def threshold_predictions(thresholded_preds, thr=0.01):
    low_values_indices = thresholded_preds < thr
    thresholded_preds[low_values_indices] = 0
    low_values_indices = thresholded_preds >= thr
    thresholded_preds[low_values_indices] = 1
    return thresholded_preds

def write_cube(data, path):
    data = np.transpose(data,[1,2,0]).astype(np.single)
    data.tofile(path)

def read_cube(input_data_path, size):
    input_cube = np.fromfile(input_data_path, dtype=np.single)
    input_cube = np.reshape(input_cube, size)
    input_cube = input_cube.transpose((2,0,1))
    return input_cube

# 读取目录
def read_path_list(path):
    file_list = os.listdir(path)
    file_name_list = [i.split(".")[0] for i in file_list]
    file_name_list = sorted(enumerate(file_name_list), key=lambda x:x[1]) 
    file_list = [file_list[i] for i in [j[0] for j in file_name_list]]
    return file_list

# 归一化
def min_max_norm(x):
    if torch.is_tensor(x) and torch.max(x) != torch.min(x):
            x = x - torch.min(x)
            x = x / torch.max(x)        
    elif np.max(x) != np.min(x):
            x = x - np.min(x)
            x = x / np.max(x)
    return x
    
# 标准化
def mea_std_norm(x):
    if torch.is_tensor(x) and torch.std(x) != 0:
            x = (x - torch.mean(x)) / torch.std(x)
    elif np.std(x) != 0:
            x = (x - np.mean(x)) / np.std(x)
    return x

class build_dataset_pretrain(Dataset):
    def __init__(self, samples_list, dataset_path, mode, 
                 input_attr_list=["seis"],index_attr_list = ["index"],
                 output_attr_list=["rgt"],output_attr_list2=["fault"],norm=None):
        self.samples_list = samples_list
        self.dataset_path = dataset_path
        self.input_attr_list = input_attr_list
        self.output_attr_list = output_attr_list
        self.output_attr_list2 = output_attr_list2
        self.index_attr_list = index_attr_list
        self.mode = mode
        
    def __len__(self):
        return len(self.samples_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        sample_file_name = self.samples_list[idx]
        sample_file_path = os.path.join(self.dataset_path, sample_file_name + ".npy")
        sample_dict = np.load(sample_file_path, allow_pickle=True).item()
        
        sample_output = {}
        
        if self.mode in ['Train', 'Valid']:
            
            # outputs
            for i, output_attr in enumerate(self.output_attr_list):
                tmp = sample_dict[output_attr].astype(np.single)
                tmp = tmp[np.newaxis,:,:]
                tmp = (min_max_norm(tmp) * 10)
                sample_output[output_attr] = tmp # rgt
                
            for i, output_attr2 in enumerate(self.output_attr_list2):
                tmp = sample_dict[output_attr2].astype(np.single)
                tmp = tmp[np.newaxis,:,:]
                sample_output[output_attr2] = tmp # unconformities
                
            ##### inputs 
            for i, input_attr in enumerate(self.input_attr_list):
                tmp = np.array(sample_dict[input_attr]).astype(np.single)
                tmp = mea_std_norm(tmp)
                sample_output[input_attr] = tmp # seismic data
            
            for i, index_attr in enumerate(self.index_attr_list):
                tmp = np.array(int(sample_file_name)).astype(np.int64)
                sample_output[index_attr] = tmp # index data
            
        sample_output["sample_file_path"] = sample_file_path
        return  sample_output

class build_dataset_finetune(Dataset):
    def __init__(self, samples_list, dataset_path, mode, 
                 input_attr_list=["seis"],index_attr_list = ["index"],
                 prior_attr_list1=["segment"],prior_attr_list2=["normal"],
                 output_attr_list=["rgt"],output_attr_list2=["fault"],norm=None):
        self.samples_list = samples_list
        self.dataset_path = dataset_path
        self.input_attr_list = input_attr_list
        self.prior_attr_list1 = prior_attr_list1
        self.prior_attr_list2 = prior_attr_list2
        self.output_attr_list = output_attr_list
        self.output_attr_list2 = output_attr_list2
        self.index_attr_list = index_attr_list
        self.mode = mode
        
    def __len__(self):
        return len(self.samples_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        sample_file_name = self.samples_list[idx]
        sample_file_path = os.path.join(self.dataset_path, sample_file_name + ".npy")
        sample_dict = np.load(sample_file_path, allow_pickle=True).item()
        
        sample_output = {}
        
        if self.mode in ['Train', 'Valid']:
            
            # outputs
            for i, output_attr in enumerate(self.output_attr_list):
                tmp = sample_dict[output_attr].astype(np.single)
                tmp = tmp[np.newaxis,:,:]
                tmp = (min_max_norm(tmp) * 10)
                sample_output[output_attr] = tmp # rgt
                
            for i, output_attr2 in enumerate(self.output_attr_list2):
                tmp = sample_dict[output_attr2].astype(np.single)
                tmp = tmp[np.newaxis,:,:]
                sample_output[output_attr2] = tmp # fault
                
            ##### inputs 
            for i, input_attr in enumerate(self.input_attr_list):
                tmp = np.array(sample_dict[input_attr]).astype(np.single)
                tmp = mea_std_norm(tmp)
                sample_output[input_attr] = tmp # seismic data
                
            for i, prior_attr1 in enumerate(self.prior_attr_list1):
                tmp = sample_dict[prior_attr1].astype(np.single)
                tmp = tmp[np.newaxis,:,:]
                sample_output[prior_attr1] = tmp # segment
                
            for i, prior_attr2 in enumerate(self.prior_attr_list2):
                tmp = sample_dict[prior_attr2].astype(np.single)
                tmp = tmp[np.newaxis,:,:]
                sample_output[prior_attr2] = tmp # normal
            
            for i, index_attr in enumerate(self.index_attr_list):
                tmp = np.array(int(sample_file_name)).astype(np.int64)
                sample_output[index_attr] = tmp # index data
            
        sample_output["sample_file_path"] = sample_file_path
        return  sample_output
    
class build_dataset_lora(Dataset):
    def __init__(self, samples_list, dataset_path, mode, 
                 input_attr_list=["seis"],index_attr_list = ["index"],
                 prior_attr_list1=["segment"],prior_attr_list2=["normal"],prior_attr_list3=["horizon"],
                 output_attr_list=["rgt"],output_attr_list2=["fault"],norm=None):
        self.samples_list = samples_list
        self.dataset_path = dataset_path
        self.input_attr_list = input_attr_list
        self.prior_attr_list1 = prior_attr_list1
        self.prior_attr_list2 = prior_attr_list2
        self.prior_attr_list3 = prior_attr_list3
        self.output_attr_list = output_attr_list
        self.output_attr_list2 = output_attr_list2
        self.index_attr_list = index_attr_list
        self.mode = mode
        
    def __len__(self):
        return len(self.samples_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        sample_file_name = self.samples_list[idx]
        sample_file_path = os.path.join(self.dataset_path, sample_file_name + ".npy")
        sample_dict = np.load(sample_file_path, allow_pickle=True).item()
        
        sample_output = {}
        
        if self.mode in ['Train', 'Valid']:
            
            # outputs
            for i, output_attr in enumerate(self.output_attr_list):
                tmp = sample_dict[output_attr].astype(np.single)
                tmp = tmp[np.newaxis,:,:]
                tmp = (min_max_norm(tmp) * 10)
                sample_output[output_attr] = tmp # rgt
                
            for i, output_attr2 in enumerate(self.output_attr_list2):
                tmp = sample_dict[output_attr2].astype(np.single)
                tmp = tmp[np.newaxis,:,:]
                sample_output[output_attr2] = tmp # fault
                
            ##### inputs 
            for i, input_attr in enumerate(self.input_attr_list):
                tmp = np.array(sample_dict[input_attr]).astype(np.single)
                tmp = mea_std_norm(tmp)
                sample_output[input_attr] = tmp # seismic data
                
            for i, prior_attr1 in enumerate(self.prior_attr_list1):
                tmp = sample_dict[prior_attr1].astype(np.single)
                tmp = tmp[np.newaxis,:,:]
                sample_output[prior_attr1] = tmp # segment
                
            for i, prior_attr2 in enumerate(self.prior_attr_list2):
                tmp = sample_dict[prior_attr2].astype(np.single)
                tmp = tmp[np.newaxis,:,:]
                sample_output[prior_attr2] = tmp # normal
                
            for i, prior_attr3 in enumerate(self.prior_attr_list3):
                tmp = sample_dict[prior_attr3].astype(np.single)
                tmp = tmp[np.newaxis,:,:]
                sample_output[prior_attr3] = tmp # horizon
            
            for i, index_attr in enumerate(self.index_attr_list):
                tmp = np.array(int(sample_file_name)).astype(np.int64)
                sample_output[index_attr] = tmp # index data
            
        sample_output["sample_file_path"] = sample_file_path
        return  sample_output
    
# SSIM Loss 
class SSIMLoss(nn.Module):
    def __init__(self, channel, filter_size):
        super(SSIMLoss, self).__init__()
        self.mssim = mssim(channel=channel, filter_size=filter_size)
    def forward(self, output, target):
        loss = (1 - self.mssim(output, target))
        return loss
    
# SILog Loss    
class SILogLoss(nn.Module):
    def __init__(self):
        super(SILogLoss, self).__init__()

    def forward(self, depth_est, depth_gt):
        g = torch.log(depth_est) - torch.log(depth_gt)
        Dg = torch.var(g) + 0.15 * torch.pow(torch.mean(g), 2)
        loss = 10 * torch.sqrt(Dg)
        return loss
    
# 曲线光滑函数
def smooth(v, w=0.85):
    last = v[0]
    smoothed = []
    for point in v:
        smoothed_val = last * w + (1 - w) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed    

####################################################################################
# 1. Pretraining Stage (self-supervised training)
####################################################################################
def petrain_stage1(param, model, train_data, input_attrs=["seis"],
                       output_attrs=["rgt"],output_attrs2=["fault"],plot=True):
    # parameters
    epochs = param['epochs']
    batch_size = param['batch_size']
    lr = param['lr']
    lr_patience = param['lr_patience']
    lr_factor = param['lr_factor']
    optimizer_type = param['optimizer_type']
    gamma = param['gamma']
    step_size = param['step_size']
    momentum = param['momentum']
    weight_decay = param['weight_decay']
    disp_inter = param['disp_inter']
    save_inter = param['save_inter']
    checkpoint_path = param['checkpoint_path']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, drop_last=True)
    if optimizer_type == "SGD":
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, momentum=momentum, weight_decay=weight_decay)
        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif optimizer_type == "Adam":
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)
        scheduler = ReduceLROnPlateau(optimizer, 'min', patience=lr_patience, factor=lr_factor)
    
    # Main cycle
    epoch_loss_train, epoch_lr = [], []
    epoch_loss_train = []
    
    best_mse = 1e50
    for epoch in range(epochs):
        # Training stage
        model.train()
        loss_train_per_epoch = 0
        
        for batch_idx, batch_samples in enumerate(train_loader):
            data = batch_samples[input_attrs[0]].to(torch.float32).unsqueeze(1)
            data = Variable(data.to(device)) # seismic data [bs,channel,X,Y]
            
            target = batch_samples[output_attrs[0]].to(torch.float32)
            target = Variable(target.to(device)) # rgt label [bs,channel,X,Y]
            target2 = batch_samples[output_attrs2[0]].to(torch.float32)
            target2 = Variable(target2.to(device)) # fault label [bs,channel,X,Y]
            indexs = batch_samples["index"].to(torch.int32)
            indexs = Variable(indexs.to(device)) # index [bs]

            optimizer.zero_grad()
            
            ### training
            loss, target_i, mask = model(data, mask_ratio=0.75)
            loss.backward()
            optimizer.step() 
            loss_train_per_epoch += loss.item()
        
        loss_train_per_epoch = loss_train_per_epoch / len(train_loader)
        epoch_loss_train.append(loss_train_per_epoch)
        epoch_lr.append(optimizer.param_groups[0]['lr'])
        
        # 保存模型
        if epoch % save_inter == 0:
            state = {'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
            filename = os.path.join(checkpoint_path, 'checkpoint-epoch{}.pth'.format(epoch))
            torch.save(state, filename)

        # 保存最优模型(step-1 and step-2)
        if loss_train_per_epoch < best_mse:
            state = {'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
            filename = os.path.join(checkpoint_path, 'checkpoint-best.pth')
            torch.save(state, filename)
            best_mse = loss_train_per_epoch

        scheduler.step(loss_train_per_epoch)
        
        # show loss
        if epoch % disp_inter == 0: 
            print('Epoch:{}, Pretrain Loss:{:.6f} Learning rate: {:.6f}'.format(epoch, loss_train_per_epoch, epoch_lr[epoch]))
            
    # Training loss curves
    if plot:
        x = [i for i in range(epochs)]
        fig = plt.figure(figsize=(12, 4))
        ax = fig.add_subplot(1, 2, 1)
        ax.plot(x, smooth(epoch_loss_train, 0.6), label='Training loss')
        ax.set_xlabel('Epoch', fontsize=15)
        ax.set_ylabel('Loss', fontsize=15)
        ax.set_title(f'Training curve', fontsize=15)
        ax.grid(True)
        plt.legend(loc='upper right', fontsize=15)

        ax = fig.add_subplot(1, 2, 2)
        ax.plot(x, epoch_lr,  label='Learning Rate')
        ax.set_xlabel('Epoch', fontsize=15)
        ax.set_ylabel('Learning Rate', fontsize=15)
        ax.set_title(f'Learning rate curve', fontsize=15)
        ax.grid(True)
        plt.legend(loc='upper right', fontsize=15)
        plt.show()
        
    logs = {"epoch_loss_train":epoch_loss_train, "epoch_lr":epoch_lr,}
    np.save(os.path.join(checkpoint_path, 'logs.npy'), logs)
    print("log has saved")
    return model

def petrain_stage1_withsyn(param, model, train_data1, train_data2, input_attrs=["seis"],
                           output_attrs=["rgt"],output_attrs2=["fault"],plot=True):
    # parameters
    epochs = param['epochs']
    batch_size = param['batch_size']
    lr = param['lr']
    lr_patience = param['lr_patience']
    lr_factor = param['lr_factor']
    optimizer_type = param['optimizer_type']
    gamma = param['gamma']
    step_size = param['step_size']
    momentum = param['momentum']
    weight_decay = param['weight_decay']
    disp_inter = param['disp_inter']
    save_inter = param['save_inter']
    checkpoint_path = param['checkpoint_path']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_loader1 = DataLoader(dataset=train_data1, batch_size=batch_size, shuffle=True, drop_last=True)
    train_loader2 = DataLoader(dataset=train_data2, batch_size=batch_size, shuffle=True, drop_last=True)
    
    if optimizer_type == "SGD":
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, momentum=momentum, weight_decay=weight_decay)
        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif optimizer_type == "Adam":
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)
        scheduler = ReduceLROnPlateau(optimizer, 'min', patience=lr_patience, factor=lr_factor)
    
    # Main cycle
    epoch_loss_train_1, epoch_lr = [], []
    epoch_loss_train_2 = []
    epoch_loss_train_sum = []
    
    best_mse = 1e50
    for epoch in range(epochs):
        # Training stage
        model.train()
        loss_train_per_epoch_1 = 0
        
        for batch_idx, batch_samples in enumerate(train_loader1):
            data = batch_samples[input_attrs[0]].to(torch.float32).unsqueeze(1)
            data = Variable(data.to(device)) # seismic data [bs,channel,X,Y]
            
            target = batch_samples[output_attrs[0]].to(torch.float32)
            target = Variable(target.to(device)) # rgt label [bs,channel,X,Y]
            target2 = batch_samples[output_attrs2[0]].to(torch.float32)
            target2 = Variable(target2.to(device)) # fault label [bs,channel,X,Y]
            indexs = batch_samples["index"].to(torch.int32)
            indexs = Variable(indexs.to(device)) # index [bs]

            optimizer.zero_grad()
            
            ### training
            loss, target_i, mask = model(data, mask_ratio=0.75)
#             print(f"fieldloss：{loss.item():.4f}")
            loss.backward()
            optimizer.step() 
            loss_train_per_epoch_1 += loss.item()
            
        # training with syndata
        model.train()
        loss_train_per_epoch_2 = 0
        
        for batch_idx, batch_samples in enumerate(train_loader2):
            data = batch_samples[input_attrs[0]].to(torch.float32).unsqueeze(1)
            data = Variable(data.to(device)) # seismic data [bs,channel,X,Y]
            
            target = batch_samples[output_attrs[0]].to(torch.float32)
            target = Variable(target.to(device)) # rgt label [bs,channel,X,Y]
            target2 = batch_samples[output_attrs2[0]].to(torch.float32)
            target2 = Variable(target2.to(device)) # fault label [bs,channel,X,Y]
            indexs = batch_samples["index"].to(torch.int32)
            indexs = Variable(indexs.to(device)) # index [bs]

            optimizer.zero_grad()
            
            ### training
            loss, target_i, mask = model(data, mask_ratio=0.75)
#             print(f"synloss：{loss.item():.4f}")
            loss.backward()
            optimizer.step() 
            loss_train_per_epoch_2 += loss.item()
        
        loss_train_per_epoch_1 = loss_train_per_epoch_1 / len(train_loader1)
        epoch_loss_train_1.append(loss_train_per_epoch_1)
        loss_train_per_epoch_2 = loss_train_per_epoch_2 / len(train_loader2)
        epoch_loss_train_2.append(loss_train_per_epoch_2)
        loss_train_per_epoch_sum = (loss_train_per_epoch_1 + loss_train_per_epoch_2)/2
        epoch_loss_train_sum.append(loss_train_per_epoch_sum)
        
        epoch_lr.append(optimizer.param_groups[0]['lr'])
        
        # 保存模型
        if epoch % save_inter == 0:
            state = {'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
            filename = os.path.join(checkpoint_path, 'checkpoint-epoch{}.pth'.format(epoch))
            torch.save(state, filename)

        # 保存最优模型(step-1 and step-2)
        if loss_train_per_epoch_sum < best_mse:
            state = {'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
            filename = os.path.join(checkpoint_path, 'checkpoint-best.pth')
            torch.save(state, filename)
            best_mse = loss_train_per_epoch_sum
        scheduler.step(loss_train_per_epoch_sum)
        
        # show loss
        if epoch % disp_inter == 0: 
            print('Epoch:{}, Pretrain Loss:{:.6f} <{:.4f} & {:.4f}> Learning rate: {:.6f}'.format(epoch, loss_train_per_epoch_sum,loss_train_per_epoch_1,loss_train_per_epoch_2, epoch_lr[epoch]))
            
    # Training loss curves
    if plot:
        x = [i for i in range(epochs)]
        fig = plt.figure(figsize=(12, 4))
        ax = fig.add_subplot(1, 2, 1)
        ax.plot(x, smooth(epoch_loss_train_sum, 0.6), label='Training loss_sum')
        ax.plot(x, smooth(epoch_loss_train_1, 0.6), label='Training loss_field')
        ax.plot(x, smooth(epoch_loss_train_2, 0.6), label='Training loss_syn')
        ax.set_xlabel('Epoch', fontsize=15)
        ax.set_ylabel('Loss', fontsize=15)
        ax.set_title(f'Training curve', fontsize=15)
        ax.grid(True)
        plt.legend(loc='upper right', fontsize=15)

        ax = fig.add_subplot(1, 2, 2)
        ax.plot(x, epoch_lr,  label='Learning Rate')
        ax.set_xlabel('Epoch', fontsize=15)
        ax.set_ylabel('Learning Rate', fontsize=15)
        ax.set_title(f'Learning rate curve', fontsize=15)
        ax.grid(True)
        plt.legend(loc='upper right', fontsize=15)
        plt.show()
        
    logs = {"epoch_loss_train_sum":epoch_loss_train_sum,"epoch_loss_train_1":epoch_loss_train_1,
            "epoch_loss_train_2":epoch_loss_train_2, "epoch_lr":epoch_lr,}
    np.save(os.path.join(checkpoint_path, 'logs.npy'), logs)
    print("log has saved")
    return model
    
####################################################################################
# 2. Supervised training Stage (supervised training)
####################################################################################

def supervised_train_stage2(param, model, train_data, valid_data, input_attrs=["seis"],
                       output_attrs=["rgt"],output_attrs2=["fault"],plot=True):
    # parameters
    epochs = param['epochs']
    batch_size = param['batch_size']
    lr = param['lr']
    lr_patience = param['lr_patience']
    lr_factor = param['lr_factor']
    optimizer_type = param['optimizer_type']
    gamma = param['gamma']
    step_size = param['step_size']
    momentum = param['momentum']
    weight_decay = param['weight_decay']
    disp_inter = param['disp_inter']
    save_inter = param['save_inter']
    checkpoint_path = param['checkpoint_path']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, drop_last=True)
    valid_loader = DataLoader(dataset=valid_data, batch_size=1, shuffle=False)
    
    if optimizer_type == "SGD":
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, momentum=momentum, weight_decay=weight_decay)
        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif optimizer_type == "Adam":
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)
        scheduler = ReduceLROnPlateau(optimizer, 'min', patience=lr_patience, factor=lr_factor)
    
    criterion1 = nn.MSELoss().to(device)
    criterion2 = SSIMLoss(channel=1,filter_size=5).to(device)
    
    # Main cycle
    epoch_loss_train, epoch_loss_valid, epoch_lr = [], [], []
    epoch_loss_train_1, epoch_loss_train_2 = [],[]
    
    best_mse = 1e50
    for epoch in range(epochs):
        # Training stage
        model.train()
        loss_train_per_epoch = 0
        loss_train_per_epoch_1 = 0
        loss_train_per_epoch_2 = 0
        
        for batch_idx, batch_samples in enumerate(train_loader):
            data = batch_samples[input_attrs[0]].to(torch.float32).unsqueeze(1)
            data = Variable(data.to(device)) # seismic data [bs,channel,X,Y]
            target = batch_samples[output_attrs[0]].to(torch.float32)
            target = Variable(target.to(device)) # rgt label [bs,channel,X,Y]
            target2 = batch_samples[output_attrs2[0]].to(torch.float32)
            target2 = Variable(target2.to(device)) # fault label [bs,channel,X,Y]
            indexs = batch_samples["index"].to(torch.int32)
            indexs = Variable(indexs.to(device)) # index [bs]
            
            optimizer.zero_grad()
            pred = model(data)
            loss1 = criterion1(pred, target)
            loss2 = 2*criterion2(pred, target)
            loss = loss1+loss2
            loss.backward()
            optimizer.step()
            
            loss_train_per_epoch_1 += loss1.item()
            loss_train_per_epoch_2 += loss2.item()
            loss_train_per_epoch += loss.item()
            
        # Validation stage
        model.eval()
        loss_valid_per_epoch = 0
        with torch.no_grad():
            for batch_idx, batch_samples in enumerate(valid_loader):
                data = batch_samples[input_attrs[0]].to(torch.float32).unsqueeze(1)
                data = Variable(data.to(device)) # seismic data [bs,channel,X,Y]
                target = batch_samples[output_attrs[0]].to(torch.float32)
                target = Variable(target.to(device)) # rgt label [bs,channel,X,Y]
                target2 = batch_samples[output_attrs2[0]].to(torch.float32)
                target2 = Variable(target2.to(device)) # fault label [bs,channel,X,Y]
                indexs = batch_samples["index"].to(torch.int32)
                indexs = Variable(indexs.to(device)) # index [bs]
                
                pred = model(data)
                loss1 = criterion1(pred, target)
                loss2 = 2*criterion2(pred, target)
                loss = loss1+loss2
                loss_valid_per_epoch += loss.item()
                
        loss_train_per_epoch = loss_train_per_epoch / len(train_loader)
        loss_valid_per_epoch = loss_valid_per_epoch / len(valid_loader)

        epoch_loss_train.append(loss_train_per_epoch)
        epoch_loss_valid.append(loss_valid_per_epoch)
        epoch_lr.append(optimizer.param_groups[0]['lr'])
        loss_train_per_epoch_1 = loss_train_per_epoch_1 / len(train_loader)
        loss_train_per_epoch_2 = loss_train_per_epoch_2 / len(train_loader)
        epoch_loss_train_1.append(loss_train_per_epoch_1)
        epoch_loss_train_2.append(loss_train_per_epoch_2)
        
        # 保存模型
        if epoch % save_inter == 0:
            state = {'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
            filename = os.path.join(checkpoint_path, 'checkpoint-epoch{}.pth'.format(epoch))
            torch.save(state, filename)

        if loss_valid_per_epoch < best_mse:
            state = {'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
            filename = os.path.join(checkpoint_path, 'checkpoint-best.pth')
            torch.save(state, filename)
            best_mse = loss_valid_per_epoch
        scheduler.step(loss_train_per_epoch)
                
        # show loss
        if epoch % disp_inter == 0: 
            print('Epoch:{}, TrainLoss:{:.6f} <{:.4f} & {:.4f}>, ValidLoss: {:.6f}, Learning rate: {:.6f}'.format(epoch, loss_train_per_epoch,loss_train_per_epoch_1,loss_train_per_epoch_2, loss_valid_per_epoch, epoch_lr[epoch]))
            
    # Training loss curves
    if plot:
        x = [i for i in range(epochs)]
        fig = plt.figure(figsize=(12, 4))
        ax = fig.add_subplot(1, 2, 1)
        ax.plot(x, smooth(epoch_loss_train, 0.6), label='Training loss')
        ax.plot(x, smooth(epoch_loss_valid, 0.6), label='Validation loss')
        ax.set_xlabel('Epoch', fontsize=15)
        ax.set_ylabel('Loss', fontsize=15)
        ax.set_title(f'Training curve', fontsize=15)
        ax.grid(True)
        plt.legend(loc='upper right', fontsize=15)

        ax = fig.add_subplot(1, 2, 2)
        ax.plot(x, epoch_lr,  label='Learning Rate')
        ax.set_xlabel('Epoch', fontsize=15)
        ax.set_ylabel('Learning Rate', fontsize=15)
        ax.set_title(f'Learning rate curve', fontsize=15)
        ax.grid(True)
        plt.legend(loc='upper right', fontsize=15)
        plt.show()
              
    logs = {"epoch_loss_train":epoch_loss_train,
            "epoch_loss_valid":epoch_loss_valid,
            "epoch_lr":epoch_lr,
            "epoch_loss1_train":epoch_loss_train_1,
            "epoch_loss2_train":epoch_loss_train_2}
    np.save(os.path.join(checkpoint_path, 'logs.npy'), logs)
    print("log has saved")
    return model

####################################################################################
# 2.2 Supervised training Stage -2 (supervised training + RL)
####################################################################################

def supervised_train_stage2_RL(param, model, train_data, valid_data, input_attrs=["seis"],
                               prior_attrs1=["segment"],prior_attrs2=["normal"],
                               output_attrs=["rgt"],output_attrs2=["fault"],plot=True):
    # parameters
    epochs = param['epochs']
    batch_size = param['batch_size']
    lr = param['lr']
    lr_patience = param['lr_patience']
    lr_factor = param['lr_factor']
    optimizer_type = param['optimizer_type']
    gamma = param['gamma']
    step_size = param['step_size']
    momentum = param['momentum']
    weight_decay = param['weight_decay']
    disp_inter = param['disp_inter']
    save_inter = param['save_inter']
    checkpoint_path = param['checkpoint_path']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, drop_last=True)
    valid_loader = DataLoader(dataset=valid_data, batch_size=1, shuffle=False)
    
    if optimizer_type == "SGD":
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, momentum=momentum, weight_decay=weight_decay)
        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif optimizer_type == "Adam":
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)
        scheduler = ReduceLROnPlateau(optimizer, 'min', patience=lr_patience, factor=lr_factor)
    
    criterion1 = nn.MSELoss().to(device)
    criterion2 = SSIMLoss(channel=1,filter_size=5).to(device)
    criterion3 = SegmentLoss().to(device)
    criterion4 = NormalLoss().to(device)
    
    # Main cycle
    epoch_loss_train, epoch_loss_valid, epoch_lr = [], [], []
    epoch_loss_train_1, epoch_loss_train_2 = [],[]
    epoch_loss_train_3, epoch_loss_train_4 = [],[]
    
    best_mse = 1e50
    for epoch in range(epochs):
        # Training stage
        model.train()
        loss_train_per_epoch = 0
        loss_train_per_epoch_1 = 0
        loss_train_per_epoch_2 = 0
        loss_train_per_epoch_3 = 0
        loss_train_per_epoch_4 = 0
        
        for batch_idx, batch_samples in enumerate(train_loader):
            data = batch_samples[input_attrs[0]].to(torch.float32).unsqueeze(1)
            data = Variable(data.to(device)) # seismic data [bs,channel,X,Y]
            
            segment = batch_samples[prior_attrs1[0]].to(torch.float32)
            segment = Variable(segment.to(device)) # segment label [bs,channel,X,Y]
            normal = batch_samples[prior_attrs2[0]].to(torch.float32)
            normal = Variable(normal.to(device)) # normal label [bs,channel,X,Y,2]
            
            target = batch_samples[output_attrs[0]].to(torch.float32)
            target = Variable(target.to(device)) # rgt label [bs,channel,X,Y]
            target2 = batch_samples[output_attrs2[0]].to(torch.float32)
            target2 = Variable(target2.to(device)) # fault label [bs,channel,X,Y]
            indexs = batch_samples["index"].to(torch.int32)
            indexs = Variable(indexs.to(device)) # index [bs]
            
            optimizer.zero_grad()
            pred = model(data)
            loss1 = criterion1(pred, target)
            loss2 = 2*criterion2(pred, target)
            loss3 = 10*criterion3(pred, segment)
            loss4 = criterion4(pred, normal)
            loss = loss1 + loss2 + loss3 + loss4
            loss.backward()
            optimizer.step()
            
            loss_train_per_epoch_1 += loss1.item()
            loss_train_per_epoch_2 += loss2.item()
            loss_train_per_epoch_3 += loss3.item()
            loss_train_per_epoch_4 += loss4.item()
            loss_train_per_epoch += loss.item()
            
        # Validation stage
        model.eval()
        loss_valid_per_epoch = 0
        with torch.no_grad():
            for batch_idx, batch_samples in enumerate(valid_loader):
                data = batch_samples[input_attrs[0]].to(torch.float32).unsqueeze(1)
                data = Variable(data.to(device)) # seismic data [bs,channel,X,Y]
                segment = batch_samples[prior_attrs1[0]].to(torch.float32)
                segment = Variable(segment.to(device)) # segment label [bs,channel,X,Y]
                normal = batch_samples[prior_attrs2[0]].to(torch.float32)
                normal = Variable(normal.to(device)) # normal label [bs,channel,X,Y,2]
                target = batch_samples[output_attrs[0]].to(torch.float32)
                target = Variable(target.to(device)) # rgt label [bs,channel,X,Y]
                target2 = batch_samples[output_attrs2[0]].to(torch.float32)
                target2 = Variable(target2.to(device)) # fault label [bs,channel,X,Y]
                indexs = batch_samples["index"].to(torch.int32)
                indexs = Variable(indexs.to(device)) # index [bs]
                
                pred = model(data)
                loss1 = criterion1(pred, target)
                loss2 = 2*criterion2(pred, target)
                loss3 = 10*criterion3(pred, segment)
                loss4 = criterion4(pred, normal)
                loss = loss1 + loss2 + loss3 + loss4
                loss_valid_per_epoch += loss.item()
                
        loss_train_per_epoch = loss_train_per_epoch / len(train_loader)
        loss_valid_per_epoch = loss_valid_per_epoch / len(valid_loader)

        epoch_loss_train.append(loss_train_per_epoch)
        epoch_loss_valid.append(loss_valid_per_epoch)
        epoch_lr.append(optimizer.param_groups[0]['lr'])
        loss_train_per_epoch_1 = loss_train_per_epoch_1 / len(train_loader)
        loss_train_per_epoch_2 = loss_train_per_epoch_2 / len(train_loader)
        loss_train_per_epoch_3 = loss_train_per_epoch_3 / len(train_loader)
        loss_train_per_epoch_4 = loss_train_per_epoch_4 / len(train_loader)
        epoch_loss_train_1.append(loss_train_per_epoch_1)
        epoch_loss_train_2.append(loss_train_per_epoch_2)
        epoch_loss_train_3.append(loss_train_per_epoch_3)
        epoch_loss_train_4.append(loss_train_per_epoch_4)
        
        # 保存模型
        if epoch % save_inter == 0:
            state = {'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
            filename = os.path.join(checkpoint_path, 'checkpoint-epoch{}.pth'.format(epoch))
            torch.save(state, filename)

        if loss_valid_per_epoch < best_mse:
            state = {'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
            filename = os.path.join(checkpoint_path, 'checkpoint-best.pth')
            torch.save(state, filename)
            best_mse = loss_valid_per_epoch
        scheduler.step(loss_train_per_epoch)
                
        # show loss
        if epoch % disp_inter == 0: 
            print('Epoch:{}, TrainLoss:{:.6f} <{:.4f} & {:.4f} & {:.4f} & {:.4f}>, ValidLoss: {:.6f}, Learning rate: {:.6f}'.format(epoch, loss_train_per_epoch,loss_train_per_epoch_1,loss_train_per_epoch_2,loss_train_per_epoch_3,loss_train_per_epoch_4,loss_valid_per_epoch, epoch_lr[epoch]))
            
    # Training loss curves
    if plot:
        x = [i for i in range(epochs)]
        fig = plt.figure(figsize=(12, 4))
        ax = fig.add_subplot(1, 2, 1)
        ax.plot(x, smooth(epoch_loss_train, 0.6), label='Training loss')
        ax.plot(x, smooth(epoch_loss_valid, 0.6), label='Validation loss')
        ax.set_xlabel('Epoch', fontsize=15)
        ax.set_ylabel('Loss', fontsize=15)
        ax.set_title(f'Training curve', fontsize=15)
        ax.grid(True)
        plt.legend(loc='upper right', fontsize=15)

        ax = fig.add_subplot(1, 2, 2)
        ax.plot(x, epoch_lr,  label='Learning Rate')
        ax.set_xlabel('Epoch', fontsize=15)
        ax.set_ylabel('Learning Rate', fontsize=15)
        ax.set_title(f'Learning rate curve', fontsize=15)
        ax.grid(True)
        plt.legend(loc='upper right', fontsize=15)
        plt.show()
              
    logs = {"epoch_loss_train":epoch_loss_train,
            "epoch_loss_valid":epoch_loss_valid,
            "epoch_lr":epoch_lr,
            "epoch_loss1_train":epoch_loss_train_1,
            "epoch_loss2_train":epoch_loss_train_2,
            "epoch_loss3_train":epoch_loss_train_3,
            "epoch_loss4_train":epoch_loss_train_4}
    np.save(os.path.join(checkpoint_path, 'logs.npy'), logs)
    print("log has saved")
    return model

####################################################################################
# 3. Reinforcement Learning Stage (unsupervised training)
####################################################################################

def rl_train_stage3(param, model, train_data1, train_data2, input_attrs=["seis"],
                               prior_attrs1=["segment"],prior_attrs2=["normal"],
                               output_attrs=["rgt"],output_attrs2=["fault"],plot=True):
    
    # parameters
    epochs = param['epochs']
    batch_size = param['batch_size']
    lr = param['lr']
    lr_patience = param['lr_patience']
    lr_factor = param['lr_factor']
    optimizer_type = param['optimizer_type']
    gamma = param['gamma']
    step_size = param['step_size']
    momentum = param['momentum']
    weight_decay = param['weight_decay']
    disp_inter = param['disp_inter']
    save_inter = param['save_inter']
    checkpoint_path = param['checkpoint_path']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_loader1 = DataLoader(dataset=train_data1, batch_size=batch_size, shuffle=True, drop_last=True)
    train_loader2 = DataLoader(dataset=train_data2, batch_size=batch_size, shuffle=True, drop_last=True)
    
    if optimizer_type == "SGD":
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, momentum=momentum, weight_decay=weight_decay)
        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif optimizer_type == "Adam":
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)
        scheduler = ReduceLROnPlateau(optimizer, 'min', patience=lr_patience, factor=lr_factor)
    
    criterion1 = nn.MSELoss().to(device)
    criterion2 = SSIMLoss(channel=1,filter_size=5).to(device)
    criterion3 = SegmentLoss().to(device)
    criterion4 = NormalLoss().to(device)
    
    # Main cycle
#     syn_ratio = 0.5
#     max_syn_batches = int(syn_ratio*len(train_loader2)/batch_size)
    max_syn_batches = 100 # 100*batch_size=1600samples for training
    epoch_loss_field, epoch_loss_syn, epoch_lr = [], [], []
    epoch_loss_all = []
    epoch_loss_field_1, epoch_loss_field_2 = [],[]
    epoch_loss_syn_1, epoch_loss_syn_2 = [],[]
    epoch_loss_syn_3, epoch_loss_syn_4 = [],[]
    
    best_mse = 1e50
    for epoch in range(epochs):
        # Training (field datasets)
        model.train()
        loss_field_per_epoch = 0
        loss_field_per_epoch_1 = 0
        loss_field_per_epoch_2 = 0
        
        for batch_idx, batch_samples in enumerate(train_loader1):
            data = batch_samples[input_attrs[0]].to(torch.float32).unsqueeze(1)
            data = Variable(data.to(device)) # seismic data [bs,channel,X,Y]
            
            segment = batch_samples[prior_attrs1[0]].to(torch.float32)
            segment = Variable(segment.to(device)) # segment label [bs,channel,X,Y]
            normal = batch_samples[prior_attrs2[0]].to(torch.float32)
            normal = Variable(normal.to(device)) # normal label [bs,channel,X,Y,2]
            
            target = batch_samples[output_attrs[0]].to(torch.float32)
            target = Variable(target.to(device)) # rgt label [bs,channel,X,Y]
            target2 = batch_samples[output_attrs2[0]].to(torch.float32)
            target2 = Variable(target2.to(device)) # fault label [bs,channel,X,Y]
            indexs = batch_samples["index"].to(torch.int32)
            indexs = Variable(indexs.to(device)) # index [bs]
            
            optimizer.zero_grad()
            pred = model(data)
            loss1 = 10*criterion3(pred, segment)
            loss2 = criterion4(pred, normal)
            loss = loss1 + loss2
#             print(loss1,loss2)
            loss.backward()
            optimizer.step()
            
            loss_field_per_epoch_1 += loss1.item()
            loss_field_per_epoch_2 += loss2.item()
            loss_field_per_epoch += loss.item()
            
        # Training (syn dataset)
        model.train()
        loss_syn_per_epoch = 0
        loss_syn_per_epoch_1 = 0
        loss_syn_per_epoch_2 = 0
        loss_syn_per_epoch_3 = 0
        loss_syn_per_epoch_4 = 0
        for batch_idx, batch_samples in enumerate(train_loader2):
            if batch_idx >= max_syn_batches:
                break
            data = batch_samples[input_attrs[0]].to(torch.float32).unsqueeze(1)
            data = Variable(data.to(device)) # seismic data [bs,channel,X,Y]

            segment = batch_samples[prior_attrs1[0]].to(torch.float32)
            segment = Variable(segment.to(device)) # segment label [bs,channel,X,Y]
            normal = batch_samples[prior_attrs2[0]].to(torch.float32)
            normal = Variable(normal.to(device)) # normal label [bs,channel,X,Y,2]

            target = batch_samples[output_attrs[0]].to(torch.float32)
            target = Variable(target.to(device)) # rgt label [bs,channel,X,Y]
            target2 = batch_samples[output_attrs2[0]].to(torch.float32)
            target2 = Variable(target2.to(device)) # fault label [bs,channel,X,Y]
            indexs = batch_samples["index"].to(torch.int32)
            indexs = Variable(indexs.to(device)) # index [bs]

            optimizer.zero_grad()
            pred = model(data)
            loss1 = criterion1(pred, target)
            loss2 = 2*criterion2(pred, target)
            loss3 = 10*criterion3(pred, segment)
            loss4 = criterion4(pred, normal)
            loss = loss1 + loss2 + loss3 + loss4 # 防模型退化
            loss.backward()
            optimizer.step()

            loss_syn_per_epoch_1 += loss1.item()
            loss_syn_per_epoch_2 += loss2.item()
            loss_syn_per_epoch_3 += loss3.item()
            loss_syn_per_epoch_4 += loss4.item()
            loss_syn_per_epoch += loss.item()
            
        loss_field_per_epoch = loss_field_per_epoch / len(train_loader1)
#         loss_syn_per_epoch = loss_syn_per_epoch / len(train_loader2)max_syn_batches
        loss_syn_per_epoch = loss_syn_per_epoch / (max_syn_batches*batch_size)
        loss_all_per_epoch = loss_field_per_epoch + 0.5*loss_syn_per_epoch
        
        epoch_loss_field.append(loss_field_per_epoch)
        epoch_loss_syn.append(loss_syn_per_epoch)
        epoch_loss_all.append(loss_all_per_epoch)
        
        epoch_lr.append(optimizer.param_groups[0]['lr'])
        loss_field_per_epoch_1 = loss_field_per_epoch_1 / len(train_loader1)
        loss_field_per_epoch_2 = loss_field_per_epoch_2 / len(train_loader1)
#         loss_syn_per_epoch_1 = loss_syn_per_epoch_1 / len(train_loader2)
#         loss_syn_per_epoch_2 = loss_syn_per_epoch_2 / len(train_loader2)
#         loss_syn_per_epoch_3 = loss_syn_per_epoch_3 / len(train_loader2)
#         loss_syn_per_epoch_4 = loss_syn_per_epoch_4 / len(train_loader2)
        loss_syn_per_epoch_1 = loss_syn_per_epoch_1 / (max_syn_batches*batch_size)
        loss_syn_per_epoch_2 = loss_syn_per_epoch_2 / (max_syn_batches*batch_size)
        loss_syn_per_epoch_3 = loss_syn_per_epoch_3 / (max_syn_batches*batch_size)
        loss_syn_per_epoch_4 = loss_syn_per_epoch_4 / (max_syn_batches*batch_size)
        epoch_loss_field_1.append(loss_field_per_epoch_1)
        epoch_loss_field_2.append(loss_field_per_epoch_2)
        epoch_loss_syn_1.append(loss_syn_per_epoch_1)
        epoch_loss_syn_2.append(loss_syn_per_epoch_2)
        epoch_loss_syn_3.append(loss_syn_per_epoch_3)
        epoch_loss_syn_4.append(loss_syn_per_epoch_4)
        
        scheduler.step(loss_field_per_epoch) # 以loss_field_per_epoch决定model进化方向
        
        # 保存模型
        if epoch % save_inter == 0:
            state = {'epoch': epoch, 'state_dict': lora.lora_state_dict(model), 'optimizer': optimizer.state_dict()}
            filename = os.path.join(checkpoint_path, 'checkpoint-epoch{}.pth'.format(epoch))
            torch.save(state, filename)

        if loss_field_per_epoch < best_mse:
            state = {'epoch': epoch, 'state_dict': lora.lora_state_dict(model), 'optimizer': optimizer.state_dict()}
            filename = os.path.join(checkpoint_path, 'checkpoint-best.pth')
            torch.save(state, filename)
            best_mse = loss_field_per_epoch
                
        # show loss
        if epoch % disp_inter == 0: 
            print('Epoch:{}, FieldLoss:{:.5f} <{:.4f}&{:.4f}>, SynLoss: {:.5f} <{:.4f}&{:.4f}&{:.4f}&{:.4f}>, Learning rate: {:.6f}'.format(epoch, loss_field_per_epoch,loss_field_per_epoch_1,loss_field_per_epoch_2,loss_syn_per_epoch,loss_syn_per_epoch_1,loss_syn_per_epoch_2,loss_syn_per_epoch_3,loss_syn_per_epoch_4,epoch_lr[epoch]))
            
    # Training loss curves
    if plot:
        x = [i for i in range(epochs)]
        fig = plt.figure(figsize=(12, 4))
        ax = fig.add_subplot(1, 2, 1)
        ax.plot(x, smooth(epoch_loss_field, 0.6), label='Field loss')
        ax.plot(x, smooth(epoch_loss_syn, 0.6), label='Syn loss')
        ax.set_xlabel('Epoch', fontsize=15)
        ax.set_ylabel('Loss', fontsize=15)
        ax.set_title(f'Training curve', fontsize=15)
        ax.grid(True)
        plt.legend(loc='upper right', fontsize=15)

        ax = fig.add_subplot(1, 2, 2)
        ax.plot(x, epoch_lr,  label='Learning Rate')
        ax.set_xlabel('Epoch', fontsize=15)
        ax.set_ylabel('Learning Rate', fontsize=15)
        ax.set_title(f'Learning rate curve', fontsize=15)
        ax.grid(True)
        plt.legend(loc='upper right', fontsize=15)
        plt.show()
              
    logs = {"epoch_loss_field":epoch_loss_field,
            "epoch_loss_syn":epoch_loss_syn,
            "epoch_lr":epoch_lr,
            "epoch_loss1_field":epoch_loss_field_1,
            "epoch_loss2_field":epoch_loss_field_2,
            "epoch_loss1_syn":epoch_loss_syn_1,
            "epoch_loss2_syn":epoch_loss_syn_2,
            "epoch_loss3_syn":epoch_loss_syn_3,
            "epoch_loss4_syn":epoch_loss_syn_4}
    np.save(os.path.join(checkpoint_path, 'logs.npy'), logs)
    print("log has saved")
    return model


    
########################################################################################################################
# Colormap
def setAlpha(cmap='jet',alpha=1, FillExceptMin=False):
    original_cmap = get_colormap(cmap)
    samples = np.linspace(0, 1, 20000) # 128 color samples
    rgba = np.array([original_cmap.map(x) for x in samples]).squeeze()
    rgba[:, -1] = alpha
    if FillExceptMin is True:
        rgba[0, -1] = 0
    cmap_new = ListedColormap(rgba)
    return cmap_new

def getStrataColors(alpha=1,FillExceptMin=False,reverse=False):
    """
    alpha: transparency of the color
    FillExceptMin : whether to fill the color at the minimum value
    """
    rgba = np.full((256,4),0).astype(np.float32)
    strata = np.array([[1.0,0.0,0.0,alpha],[1.0,0.5019608,0.0,alpha],[1.0,1.0,0.0,alpha],
                       [0.0,1,0.0,alpha],[0.0,0.5019608,0.0,alpha],[0.0,0.2509804,0.0,alpha],
                       [0,1.0,1.0,alpha],[0.0,0.5019608,1.0,alpha],[0.0,0.0,1.0,alpha],
                       [0.0,0.0,0.627451,alpha],[0.0,0.5019608,0.7529412,alpha],[1.0,0.5019608,0.5019608,alpha],
                       [0.5019608,0.5019608,1.0,alpha],[0.5019608,0.0,1.0,alpha],[0.5019608,0,0.5019608,alpha],
                       [1.0,0.5019608,1.0,alpha],[1.0,0.0,1.0,alpha],[0.5019608,0.2509804,0,alpha],
                       [0.5019608,0.5019608,0.5019608,alpha],[0.7529412,0.7529412,0.7529412,alpha],[0.2509804,0,0.2509804,alpha],
                       [0.90588236,0.7294118,0.19607843,alpha],[0.44313726,0.58431375,0.58431375,alpha],[0.5254902,0.42352942,0.4862745,alpha],
                       [0.7176471,0.54509807,0.44313726,alpha],[0.5019608,0.5019608,0,alpha],[0.7529412,0.7294118,0.8784314,alpha],
                       [0.61960787,0.85882354,0.9882353,alpha],[0.7372549,0.25882354,0.24705882,alpha],[0.8862745,0.8509804,0.627451,alpha],
                       [0.60784316,0.9411765,0.7490196,alpha],[0.62352943,0.79607844,0.105882354,alpha]])
    for i in range(32):
        for k in range(8):
            rgba[i*8+k] = strata[i]

    if FillExceptMin is True:
        rgba[0, -1] = 0
    if reverse is True:
        rgba = np.flip(rgba,axis=0)
    stratamap = ListedColormap(rgba)
    
    return stratamap

def getContrastColors(alpha=1.0,):
    """
    生成高对比度的colormap，对应Java代码的实现
    :param alpha: 透明度 (0.0-1.0)
    :return: matplotlib的LinearSegmentedColormap对象
    """
    # 10个锚点颜色: deep blue -> blue -> cyan -> green -> yellow-green -> yellow -> orange -> red -> magenta -> purple
    anchors = [
        (0.0, 0.0, 0.5),   # deep blue
        (0.0, 0.0, 1.0),   # blue
        (0.0, 1.0, 1.0),   # cyan
        (0.0, 0.8, 0.0),   # green
        (0.5, 1.0, 0.0),   # yellow-green
        (1.0, 1.0, 0.0),   # yellow
        (1.0, 0.5, 0.0),   # orange
        (1.0, 0.0, 0.0),   # red
        (1.0, 0.0, 0.5),   # magenta
        (0.5, 0.0, 0.5)    # purple
    ]
    positions = np.linspace(0, 1, len(anchors))
    red = []
    green = []
    blue = []
    for pos, (r, g, b) in zip(positions, anchors):
        red.append((pos, r, r))
        green.append((pos, g, g))
        blue.append((pos, b, b))
    cdict = {
        'red': red,
        'green': green,
        'blue': blue,
        'alpha': [(0.0, alpha, alpha), (1.0, alpha, alpha)]  # 透明度
    }
    cmap = LinearSegmentedColormap('high_contrast', cdict, N=256)
    
    return cmap

def getPetrelColors(alpha=1.0, reverse=False, name="Petrel"):
    cdict = {'red': [(0.0, 0.6313725490196078, 0.6313725490196078),(0.33, 0.0, 0.0),
            (0.4, 0.30196078431372547, 0.30196078431372547),(0.5, 0.8, 0.8),
            (0.6, 0.3803921568627451, 0.3803921568627451),
            (0.67, 0.7490196078431373, 0.7490196078431373),(1.0, 1.0, 1.0)],
        'green': [(0.0, 1.0, 1.0),(0.33, 0.0, 0.0),
            (0.4, 0.30196078431372547, 0.30196078431372547),(0.5,  0.8, 0.8),
            (0.6, 0.27058823529411763, 0.27058823529411763),
            (0.67,0.0, 0.0),(1.0,  1.0, 1.0)],
        'blue': [(0.0, 1.0, 1.0),(0.33, 0.7490196078431373, 0.7490196078431373),
            (0.4, 0.30196078431372547, 0.30196078431372547),
            (0.5, 0.8, 0.8),(0.6, 0.0, 0.0),(0.67, 0.0, 0.0),(1.0, 0.0, 0.0)],}
    # add alpha channel
    cdict["alpha"] = [(x, alpha, alpha) for x, _, _ in cdict["red"]]
    cmap = LinearSegmentedColormap(name, cdict, N=256)
    if reverse:
        cmap = cmap.reversed()
    return cmap

def num2color(values, cmap,vmin,vmax):
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = mpl.cm.get_cmap(cmap)
    return [cmap(norm(val)) for val in values]