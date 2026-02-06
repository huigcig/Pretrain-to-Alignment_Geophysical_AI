import os
import math
import random
import numpy as np
import random
import torch
from collections import deque
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean

def compute_SegmentLoss(input,segment):
    segloss = 0
    for bi in range(input.shape[0]):
        pred_flat = input[bi,0].reshape(-1)
        seg_flat = segment[bi,0].reshape(-1)
        valid_mask = seg_flat > 0
        pred_valid = pred_flat[valid_mask]
        seg_valid  = seg_flat[valid_mask].long()
        mean_per_seg = scatter_mean(pred_valid, seg_valid, dim=0) # 每个segments计算一个均值
        mean_at_pixel = mean_per_seg[seg_valid] # 传回均值
        loss_bi = (pred_valid - mean_at_pixel).abs() # L1 计算
    #     loss_bi = loss_bi / (mean_at_pixel.abs() + 1e-8) # 权重，更专注于浅层
    #     loss_bi *= seg_valid # 权重，考虑是否参考segments，前面的更靠谱
        segloss += loss_bi.mean()
    segloss = torch.sum(segloss) / input.shape[0]
    return segloss
  
class SegmentLoss(nn.Module): 
    """
    This self-supervised loss function is defined by Hui Gao (2026.1.1)
    """
    def __init__(self):
        super(SegmentLoss, self).__init__()
        
    def forward(self, input, segment):
        loss = compute_SegmentLoss(input, segment)
        return loss

    
def calculate_normal_loss(input, normal, eps=1e-20):
    """
    normal loss
    eps_tensor: avoid division by zero
    """
    bs, _, h, w = input.size()
    
    # 1-cos_similarity loss
    gx = torch.zeros_like(normal) 
    [gx[:,:,:,:,0],gx[:,:,:,:,1]] = torch.gradient(input,dim=(2,3))
    cosloss = nn.CosineSimilarity(dim=4, eps=eps)
    loss = 1 - cosloss(gx,normal)
    return torch.sum(loss)/bs/h/w

class NormalLoss(nn.Module): 
    """
    This self-supervised loss function is defined by Hui Gao (2024.10.23)
    1. Using the predicted RGT label to calculate gradient
    2. Multiply the normal vector with gradient
    3. Minimum the dot value (loss)
    """
    def __init__(self):
        super(NormalLoss, self).__init__()
        
    def forward(self, input, normal):
        """
        input: (batch_size,channel,x,y)
        shear: (batch_size,channel,x,y,2) two vectors in x&y
        """
        loss = calculate_normal_loss(input, normal)
        return loss

    
    
    