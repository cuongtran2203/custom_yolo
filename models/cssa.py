import torch
import unittest
import random
from torchinfo import summary
import os
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import xmltodict
from PIL import Image
import tqdm

import torchvision.transforms.functional as F


# coming out of adaptiveavg2d we want to have 1x1xC, this allows us to happen

class PermuteBlock(torch.nn.Module):
    def __init__(self, channels):
      super().__init__()
      self.shape = channels

    def forward(self, x):
        return torch.permute(x, self.shape)
    
    
class ECABlock(torch.nn.Module):
  def __init__(self, kernel_size=3, channel_first=None):
    super().__init__()

    self.channel_first = channel_first

    self.GAP = torch.nn.AdaptiveAvgPool2d(1)
    self.f = torch.nn.Conv1d(1, 1, kernel_size=kernel_size, padding = kernel_size // 2, bias=False)
    self.sigmoid = torch.nn.Sigmoid()


  def forward(self, x):

    x = self.GAP(x)

    # need to squeeze 4d tensor to 3d & transpose so convolution happens correctly
    x = x.squeeze(-1).transpose(-1, -2)
    x = self.f(x)
    x = x.transpose(-1, -2).unsqueeze(-1) # return to correct shape, reverse ops

    x = self.sigmoid(x)

    return x
class ChannelSwitching(torch.nn.Module):
  def __init__(self, switching_thresh):
    super().__init__()
    self.k = switching_thresh

  def forward(self, x, x_prime, w):

    self.mask = w < self.k
     # If self.mask is True, take from x_prime; otherwise, keep x's value
    x = torch.where(self.mask, x_prime, x)

    return x
class SpatialAttention(torch.nn.Module):

  def __init__(self):
    super().__init__()

    self.sigmoid = torch.nn.Sigmoid()

  def forward(self, rgb_feats, ir_feats):
    # get shape
    B, C, H, W = rgb_feats.shape

    # channel concatenation (x_cat -> B,2C,H,W)
    x_cat = torch.cat((rgb_feats, ir_feats), axis=1)

    # create w_avg attention map (w_avg -> B,1,H,W)
    cap = torch.mean(x_cat, dim=1)
    w_avg = self.sigmoid(cap)
    w_avg = w_avg.unsqueeze(1)

    # create w_max attention maps (w_max -> B,1,H,W)
    cmp = torch.max(x_cat, dim=1)[0]
    w_max = self.sigmoid(cmp)
    w_max = w_max.unsqueeze(1)

    # weighted feature map (x_cat_w -> B,2C,H,W)
    x_cat_w = x_cat * w_avg * w_max

    # split weighted feature map (x_ir_w, x_rgb_w -> B,C,H,W)
    x_rgb_w = x_cat_w[:,:C,:,:]
    x_ir_w = x_cat_w[:,C:,:,:]

    # fuse feature maps (x_fused -> B,H,W,C)
    x_fused = (x_ir_w + x_rgb_w)/2

    return x_fused

class CSSA(torch.nn.Module):

  def __init__(self, switching_thresh=0.5, kernel_size=3, channel_first=None):
    super().__init__()

    # self.eca = ECABlock(kernel_size=kernel_size, channel_first=channel_first)
    self.eca_rgb = ECABlock(kernel_size=kernel_size, channel_first=channel_first)
    self.eca_ir = ECABlock(kernel_size=kernel_size, channel_first=channel_first)
    self.cs = ChannelSwitching(switching_thresh=switching_thresh)
    self.sa = SpatialAttention()

  def forward(self, rgb_input, ir_input):
    # channel switching for RGB input
    rgb_w = self.eca_rgb(rgb_input)
    rgb_feats = self.cs(rgb_input, ir_input, rgb_w)

    # channel switching for IR input
    ir_w = self.eca_ir(ir_input)
    ir_feats = self.cs(ir_input, rgb_input, ir_w)

    # spatial attention
    fused_feats = self.sa(rgb_feats, ir_feats)
    b,c,h,w = fused_feats.size()
    # fused_feats = torch.reshape(fused_feats,[b,h,w,c])

    return fused_feats
