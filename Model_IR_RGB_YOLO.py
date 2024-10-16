'''
Module này định nghĩa ra model phát hiện đối tượng dựa trên input là ảnh RGB và ảnh IR  với kiến trúc như sau:


Backbone_YOLO_RGB
                + CSSA module + Head_2 -> OUTPUT
Backbone_YOLO_IR

'''
import torch
import torch.nn as nn
from common import *
import yaml
from models.csp_darknet import YOLOv5CSPDarknet
from models.yolov5_pafpn import YOLOv5PAFPN
from models.yolov5_head import YOLOv5HeadModule,YOLOv5Head
from models.cssa import CSSA
from models.dense_heads.yolov5_head import *
import numpy as np
from typing import List
import megengine.functional as F
import megengine.module as M
from process_coupled_head import *


def meshgrid(x, y):
    """meshgrid wrapper for megengine"""
    assert len(x.shape) == 1
    assert len(y.shape) == 1
    mesh_shape = (y.shape[0], x.shape[0])
    mesh_x = F.broadcast_to(x, mesh_shape)
    mesh_y = F.broadcast_to(y.reshape(-1, 1), mesh_shape)
    return mesh_x, mesh_y

def autopad(k, p=None, d=1):
    """
    Pads kernel to 'same' output shape, adjusting for optional dilation; returns padding size.

    `k`: kernel, `p`: padding, `d`: dilation.
    """
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    """Applies a convolution, batch normalization, and activation function to an input tensor in a neural network."""

    default_act = nn.ReLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act="ReLU"):
        """Initializes a standard convolution layer with optional batch normalization and activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act =="ReLU" else  nn.SiLU()

    def forward(self, x):
        """Applies a convolution followed by batch normalization and an activation function to the input tensor `x`."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Applies a fused convolution and activation function to the input tensor `x`."""
        return self.act(self.conv(x))


class Decoupled_Head(nn.Module):
  def __init__(self,in_channel,outchannels:List):
      super().__init__()
      self.conv1 = Conv(in_channel,256,1,act="SiLU")
      #Coupled 1
      self.conv2 = Conv(256,256,3,act="SiLU")
      self.conv3 = Conv(256,256,3,act="SiLU")
      #Coupled 2
      self.conv4 = Conv(256,256,3,act="SiLU")
      self.conv5 = Conv(256,256,3,act="SiLU")
      # Layer Output
      self.out1 = Conv(256,outchannels[0],1,act="SiLU")
      self.out2 = Conv(256,outchannels[1],1,act="SiLU")
      self.out3 = Conv(256,outchannels[2],1,act="SiLU")

  def forward(self,input_tensor):
    x = self.conv1(input_tensor)
    #Coupled 1
    x1 = self.conv2(x)
    x2 = self.conv3(x1)
    #Coupled 2
    x3 = self.conv4(x)
    x4 = self.conv5(x3)
    #Return output
    cls = torch.sigmoid(self.out1(x2))
    reg = self.out2(x4)
    obj = torch.sigmoid(self.out3(x4))
    # print(cls.shape)
    # print(reg.shape)
    # print(obj.shape)
    output = torch.cat([reg,obj,cls],dim=1)
    return output
    
    
    
      
      
class Coupled_Head(nn.Module):
  def __init__(self,in_channel=None,outchannel=None):
    super().__init__()
    self.conv1x1 = Conv(in_channel,in_channel,1,act="ReLU")
    self.conv3x3 = Conv(in_channel,outchannel,3,act="ReLU")
  def forward(self,x):
    x = self.conv1x1(x)
    x = self.conv3x3(x)
    return x
     
     
      

# class Custom_Head_YOLOv5(nn.Module):
#     def __init__(self,in_channels=[256,512,1024],num_classes=3,anchors=3):
#         super().__init__()
#         self.conv1 = nn.Conv2d(in_channels=in_channels[0],out_channels=)
class IR_RGB_Model(nn.Module):
    def __init__(self,num_classes,anchors=None):
        super().__init__()
        self.RGB_backbone = YOLOv5CSPDarknet()
        self.IR_backbone = YOLOv5CSPDarknet()
        self.channel_sizes = [256,512,1024]
        self.cssa_switching_thresh = 0.5
        self.anchors = anchors
        self.strides = [8, 16, 32]
        self.grids = [F.zeros(1)] * 3
        # ECA kernel sizes
        ECA_kernels = [self.find_ECA_k(channel) for channel in self.channel_sizes]
        self.CSSA_1 = CSSA(self.cssa_switching_thresh,ECA_kernels[0])
        self.CSSA_2 = CSSA(self.cssa_switching_thresh,ECA_kernels[1])
        self.CSSA_3 = CSSA(self.cssa_switching_thresh,ECA_kernels[2])
        self.neck = YOLOv5PAFPN(in_channels=[256,512,1024],out_channels=[256,512,1024])
        # self.head = YOLOv5HeadModule(num_classes=3,in_channels=[256,512,1024])
        if self.anchors is not None:
          #Predict score
          self.head1 = Coupled_Head(self.channel_sizes[0],anchors*(num_classes+5))
          self.head2 = Coupled_Head(self.channel_sizes[1],anchors*(num_classes+5))
          self.head3 = Coupled_Head(self.channel_sizes[2],anchors*(num_classes+5))
        else:
          # Predict by decouple Head
          self.head1 = Decoupled_Head(self.channel_sizes[0],[num_classes,4,1])
          self.head2 = Decoupled_Head(self.channel_sizes[1],[num_classes,4,1])
          self.head3 = Decoupled_Head(self.channel_sizes[2],[num_classes,4,1])
          
          
          
        
    def find_ECA_k(self, channel):

      gamma, beta = 2, 1

      k = int(abs(( np.log2(channel) / 2) + (beta/gamma) ))

      if k % 2 == 0:
        k -= 1

      return k
    def get_output_and_grid(self, output, k, stride, dtype):
      grid = self.grids[k]

      batch_size = output.shape[0]
      n_ch = 5 + self.num_classes
      hsize, wsize = output.shape[-2:]
      if grid.shape[2:4] != output.shape[2:4]:
          yv, xv = meshgrid([torch.arange(hsize), torch.arange(wsize)])
          grid = torch.stack((xv, yv), 2).view(1, 1, hsize, wsize, 2).type(dtype)
          self.grids[k] = grid

      output = output.view(batch_size, 1, n_ch, hsize, wsize)
      output = output.permute(0, 1, 3, 4, 2).reshape(
          batch_size, hsize * wsize, -1
      )
      grid = grid.view(1, -1, 2).float()
      output[..., :2] = (output[..., :2] + grid) * stride
      output[..., 2:4] = torch.exp(output[..., 2:4]) * stride
      return output, grid

    def decode_outputs(self, outputs):
        grids = []
        strides = []
        for (hsize, wsize), stride in zip(self.hw, self.strides):
            xv, yv = meshgrid(F.arange(hsize), F.arange(wsize))
            grid = F.stack((xv, yv), 2).reshape(1, -1, 2)
            grids.append(grid)
            shape = grid.shape[:2]
            strides.append(F.full((*shape, 1), stride))

        grids = F.concat(grids, axis=1)
        strides = F.concat(strides, axis=1)

        outputs[..., :2] = (outputs[..., :2] + grids) * strides
        outputs[..., 2:4] = F.exp(outputs[..., 2:4]) * strides
        return outputs
    def forward(self,rgb,ir,inference_mode=False):
        rgb_features = self.RGB_backbone(rgb)
        ir_features = self.IR_backbone(ir)
        c = [self.CSSA_1(rgb_features[0],ir_features[0]),
             self.CSSA_2(rgb_features[1],ir_features[1]),
             self.CSSA_3(rgb_features[2],ir_features[2])]
        # for c_ in c:
        #     print(c_.shape)
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        features_neck = self.neck(c)
        if self.anchors is not None:
          out1, out2, out3 = self.head1(features_neck[0]),self.head2(features_neck[1]),self.head3(features_neck[2])
          outputs = [out1,out2,out3]
          outputs = torch.concat([torch.flatten(x, start_dim=2) for x in outputs], dim=2)
          outputs = torch.tensor(outputs.detach().cpu().numpy().transpose(0,2,1),dtype=torch.float32).to(self.device)
          return outputs
        else:
          out1, out2, out3 = self.head1(features_neck[0]), self.head2(features_neck[1]), self.head3(features_neck[2])
          outputs = [out1,out2,out3]
          self.hw = [x.shape[-2:] for x in outputs]
          # [batch, n_anchors_all, 85]
          outputs = torch.concat([torch.flatten(x, start_dim=2) for x in outputs], dim=2)
          outputs = torch.tensor(outputs.detach().cpu().numpy().transpose(0,2,1),dtype=torch.float32).to(self.device)
          if inference_mode:
            return self.decode_outputs(outputs.detach().cpu().numpy())
          else:
            return outputs
         
    
if __name__ == "__main__":
    model = IR_RGB_Model(num_classes=3,anchors=1)
    input_rgb = torch.rand((1,3,640,640))
    input_ir = torch.rand((1,3,640,640))
    outs = model(input_rgb,input_ir)
    # output = non_max_suppression(outs)
    # print(output[0].shape)
        
        
        