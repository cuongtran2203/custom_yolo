'''
Module này định nghĩa ra model phát hiện đối tượng dựa trên input là ảnh RGB và ảnh IR  với kiến trúc như sau:


Backbone_YOLO_RGB
                + CSSA module + Head_2 -> OUTPUT
Backbone_YOLO_IR

'''
import torch
import torch.nn as nn
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
from process import *
from logging import error
from loss import IOUloss
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
    def __init__(self,num_classes,anchors=None,training=False,decode_in_inference=False):
        super().__init__()
        self.training=False
        self.RGB_backbone = YOLOv5CSPDarknet()
        self.IR_backbone = YOLOv5CSPDarknet()
        self.channel_sizes = [256,512,1024]
        self.cssa_switching_thresh = 0.5
        self.anchors = anchors
        self.decode_in_inference = decode_in_inference
        self.use_l1 = False
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
        self.use_l1 = False
        self.l1_loss = nn.L1Loss(reduction="none")
        self.bcewithlog_loss = nn.BCEWithLogitsLoss(reduction="none")
        self.iou_loss = IOUloss(reduction="none")
          
          
        
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

    def decode_outputs(self, outputs, dtype):
      grids = []
      strides = []
      for (hsize, wsize), stride in zip(self.hw, self.strides):
          yv, xv = meshgrid([torch.arange(hsize), torch.arange(wsize)])
          grid = torch.stack((xv, yv), 2).view(1, -1, 2)
          grids.append(grid)
          shape = grid.shape[:2]
          strides.append(torch.full((*shape, 1), stride))

      grids = torch.cat(grids, dim=1).type(dtype)
      strides = torch.cat(strides, dim=1).type(dtype)

      outputs = torch.cat([
          (outputs[..., 0:2] + grids) * strides,
          torch.exp(outputs[..., 2:4]) * strides,
          outputs[..., 4:]
      ], dim=-1)
      return outputs
    def forward(self,rgb,ir,labels=None,imgs=None):
      
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
          outputs = []
          origin_preds = []
          x_shifts = []
          y_shifts = []
          expanded_strides = []
          out1, out2, out3 = self.head1(features_neck[0]), self.head2(features_neck[1]), self.head3(features_neck[2])
          
          if self.training:
            for k, output,stride_this_level in enumerate(zip(outputs,self.strides)):
              output, grid = self.get_output_and_grid(
                      output, k, stride_this_level, rgb.type()
                  )
              x_shifts.append(grid[:, :, 0])
              y_shifts.append(grid[:, :, 1])
              expanded_strides.append(
                      torch.zeros(1, grid.shape[1])
                      .fill_(stride_this_level)
                      .type_as(rgb)       
                  )
              outputs.append(output)
          else:
            outputs = [out1,out2,out3]
          if self.training:
              return self.get_losses(
                imgs,
                x_shifts,
                y_shifts,
                expanded_strides,
                labels,
                torch.cat(outputs, 1),
                origin_preds,
                dtype=rgb.dtype,
            )
          else:
              self.hw = [x.shape[-2:] for x in outputs]
              # [batch, n_anchors_all, 85]
              outputs = torch.cat(
                  [x.flatten(start_dim=2) for x in outputs], dim=2
              ).permute(0, 2, 1)
              if self.decode_in_inference:
                  return self.decode_outputs(outputs, dtype=rgb.type())
              else:
                  return outputs
        
        
    def get_losses(
      self,
      imgs,
      x_shifts,
      y_shifts,
      expanded_strides,
      labels,
      outputs,
      origin_preds,
      dtype,
      ):
      bbox_preds = outputs[:, :, :4]  # [batch, n_anchors_all, 4]
      obj_preds = outputs[:, :, 4:5]  # [batch, n_anchors_all, 1]
      cls_preds = outputs[:, :, 5:]  # [batch, n_anchors_all, n_cls]

      # calculate targets
      nlabel = (labels.sum(dim=2) > 0).sum(dim=1)  # number of objects

      total_num_anchors = outputs.shape[1]
      x_shifts = torch.cat(x_shifts, 1)  # [1, n_anchors_all]
      y_shifts = torch.cat(y_shifts, 1)  # [1, n_anchors_all]
      expanded_strides = torch.cat(expanded_strides, 1)
      if self.use_l1:
          origin_preds = torch.cat(origin_preds, 1)

      cls_targets = []
      reg_targets = []
      l1_targets = []
      obj_targets = []
      fg_masks = []

      num_fg = 0.0
      num_gts = 0.0

      for batch_idx in range(outputs.shape[0]):
          num_gt = int(nlabel[batch_idx])
          num_gts += num_gt
          if num_gt == 0:
              cls_target = outputs.new_zeros((0, self.num_classes))
              reg_target = outputs.new_zeros((0, 4))
              l1_target = outputs.new_zeros((0, 4))
              obj_target = outputs.new_zeros((total_num_anchors, 1))
              fg_mask = outputs.new_zeros(total_num_anchors).bool()
          else:
              gt_bboxes_per_image = labels[batch_idx, :num_gt, 1:5]
              gt_classes = labels[batch_idx, :num_gt, 0]
              bboxes_preds_per_image = bbox_preds[batch_idx]

              try:
                  (
                      gt_matched_classes,
                      fg_mask,
                      pred_ious_this_matching,
                      matched_gt_inds,
                      num_fg_img,
                  ) = self.get_assignments(  # noqa
                      batch_idx,
                      num_gt,
                      gt_bboxes_per_image,
                      gt_classes,
                      bboxes_preds_per_image,
                      expanded_strides,
                      x_shifts,
                      y_shifts,
                      cls_preds,
                      obj_preds,
                  )
              except RuntimeError as e:
                  # TODO: the string might change, consider a better way
                  if "CUDA out of memory. " not in str(e):
                      raise  # RuntimeError might not caused by CUDA OOM

                  error(
                      "OOM RuntimeError is raised due to the huge memory cost during label assignment. \
                          CPU mode is applied in this batch. If you want to avoid this issue, \
                          try to reduce the batch size or image size."
                  )
                  torch.cuda.empty_cache()
                  (
                      gt_matched_classes,
                      fg_mask,
                      pred_ious_this_matching,
                      matched_gt_inds,
                      num_fg_img,
                  ) = self.get_assignments(  # noqa
                      batch_idx,
                      num_gt,
                      gt_bboxes_per_image,
                      gt_classes,
                      bboxes_preds_per_image,
                      expanded_strides,
                      x_shifts,
                      y_shifts,
                      cls_preds,
                      obj_preds,
                      "cpu",
                  )

              torch.cuda.empty_cache()
              num_fg += num_fg_img

              cls_target = F.one_hot(
                  gt_matched_classes.to(torch.int64), self.num_classes
              ) * pred_ious_this_matching.unsqueeze(-1)
              obj_target = fg_mask.unsqueeze(-1)
              reg_target = gt_bboxes_per_image[matched_gt_inds]
              if self.use_l1:
                  l1_target = self.get_l1_target(
                      outputs.new_zeros((num_fg_img, 4)),
                      gt_bboxes_per_image[matched_gt_inds],
                      expanded_strides[0][fg_mask],
                      x_shifts=x_shifts[0][fg_mask],
                      y_shifts=y_shifts[0][fg_mask],
                  )

          cls_targets.append(cls_target)
          reg_targets.append(reg_target)
          obj_targets.append(obj_target.to(dtype))
          fg_masks.append(fg_mask)
          if self.use_l1:
              l1_targets.append(l1_target)

      cls_targets = torch.cat(cls_targets, 0)
      reg_targets = torch.cat(reg_targets, 0)
      obj_targets = torch.cat(obj_targets, 0)
      fg_masks = torch.cat(fg_masks, 0)
      if self.use_l1:
          l1_targets = torch.cat(l1_targets, 0)

      num_fg = max(num_fg, 1)
      loss_iou = (
          self.iou_loss(bbox_preds.view(-1, 4)[fg_masks], reg_targets)
      ).sum() / num_fg
      loss_obj = (
          self.bcewithlog_loss(obj_preds.view(-1, 1), obj_targets)
      ).sum() / num_fg
      loss_cls = (
          self.bcewithlog_loss(
              cls_preds.view(-1, self.num_classes)[fg_masks], cls_targets
          )
      ).sum() / num_fg
      if self.use_l1:
          loss_l1 = (
              self.l1_loss(origin_preds.view(-1, 4)[fg_masks], l1_targets)
          ).sum() / num_fg
      else:
          loss_l1 = 0.0

      reg_weight = 5.0
      loss = reg_weight * loss_iou + loss_obj + loss_cls + loss_l1

      return (
          loss,
          reg_weight * loss_iou,
          loss_obj,
          loss_cls,
          loss_l1,
          num_fg / max(num_gts, 1),
      )

    def get_l1_target(self, l1_target, gt, stride, x_shifts, y_shifts, eps=1e-8):
        l1_target[:, 0] = gt[:, 0] / stride - x_shifts
        l1_target[:, 1] = gt[:, 1] / stride - y_shifts
        l1_target[:, 2] = torch.log(gt[:, 2] / stride + eps)
        l1_target[:, 3] = torch.log(gt[:, 3] / stride + eps)
        return l1_target

    @torch.no_grad()
    def get_assignments(
        self,
        batch_idx,
        num_gt,
        gt_bboxes_per_image,
        gt_classes,
        bboxes_preds_per_image,
        expanded_strides,
        x_shifts,
        y_shifts,
        cls_preds,
        obj_preds,
        mode="gpu",
    ):

        if mode == "cpu":
            print("-----------Using CPU for the Current Batch-------------")
            gt_bboxes_per_image = gt_bboxes_per_image.cpu().float()
            bboxes_preds_per_image = bboxes_preds_per_image.cpu().float()
            gt_classes = gt_classes.cpu().float()
            expanded_strides = expanded_strides.cpu().float()
            x_shifts = x_shifts.cpu()
            y_shifts = y_shifts.cpu()

        fg_mask, geometry_relation = self.get_geometry_constraint(
            gt_bboxes_per_image,
            expanded_strides,
            x_shifts,
            y_shifts,
        )

        bboxes_preds_per_image = bboxes_preds_per_image[fg_mask]
        cls_preds_ = cls_preds[batch_idx][fg_mask]
        obj_preds_ = obj_preds[batch_idx][fg_mask]
        num_in_boxes_anchor = bboxes_preds_per_image.shape[0]

        if mode == "cpu":
            gt_bboxes_per_image = gt_bboxes_per_image.cpu()
            bboxes_preds_per_image = bboxes_preds_per_image.cpu()

        pair_wise_ious = bboxes_iou(gt_bboxes_per_image, bboxes_preds_per_image, False)

        gt_cls_per_image = (
            F.one_hot(gt_classes.to(torch.int64), self.num_classes)
            .float()
        )
        pair_wise_ious_loss = -torch.log(pair_wise_ious + 1e-8)

        if mode == "cpu":
            cls_preds_, obj_preds_ = cls_preds_.cpu(), obj_preds_.cpu()

        with torch.cuda.amp.autocast(enabled=False):
            cls_preds_ = (
                cls_preds_.float().sigmoid_() * obj_preds_.float().sigmoid_()
            ).sqrt()
            pair_wise_cls_loss = F.binary_cross_entropy(
                cls_preds_.unsqueeze(0).repeat(num_gt, 1, 1),
                gt_cls_per_image.unsqueeze(1).repeat(1, num_in_boxes_anchor, 1),
                reduction="none"
            ).sum(-1)
        del cls_preds_

        cost = (
            pair_wise_cls_loss
            + 3.0 * pair_wise_ious_loss
            + float(1e6) * (~geometry_relation)
        )

        (
            num_fg,
            gt_matched_classes,
            pred_ious_this_matching,
            matched_gt_inds,
        ) = self.simota_matching(cost, pair_wise_ious, gt_classes, num_gt, fg_mask)
        del pair_wise_cls_loss, cost, pair_wise_ious, pair_wise_ious_loss

        if mode == "cpu":
            gt_matched_classes = gt_matched_classes.cuda()
            fg_mask = fg_mask.cuda()
            pred_ious_this_matching = pred_ious_this_matching.cuda()
            matched_gt_inds = matched_gt_inds.cuda()

        return (
            gt_matched_classes,
            fg_mask,
            pred_ious_this_matching,
            matched_gt_inds,
            num_fg,
        )

    def get_geometry_constraint(
        self, gt_bboxes_per_image, expanded_strides, x_shifts, y_shifts,
    ):
        """
        Calculate whether the center of an object is located in a fixed range of
        an anchor. This is used to avert inappropriate matching. It can also reduce
        the number of candidate anchors so that the GPU memory is saved.
        """
        expanded_strides_per_image = expanded_strides[0]
        x_centers_per_image = ((x_shifts[0] + 0.5) * expanded_strides_per_image).unsqueeze(0)
        y_centers_per_image = ((y_shifts[0] + 0.5) * expanded_strides_per_image).unsqueeze(0)

        # in fixed center
        center_radius = 1.5
        center_dist = expanded_strides_per_image.unsqueeze(0) * center_radius
        gt_bboxes_per_image_l = (gt_bboxes_per_image[:, 0:1]) - center_dist
        gt_bboxes_per_image_r = (gt_bboxes_per_image[:, 0:1]) + center_dist
        gt_bboxes_per_image_t = (gt_bboxes_per_image[:, 1:2]) - center_dist
        gt_bboxes_per_image_b = (gt_bboxes_per_image[:, 1:2]) + center_dist

        c_l = x_centers_per_image - gt_bboxes_per_image_l
        c_r = gt_bboxes_per_image_r - x_centers_per_image
        c_t = y_centers_per_image - gt_bboxes_per_image_t
        c_b = gt_bboxes_per_image_b - y_centers_per_image
        center_deltas = torch.stack([c_l, c_t, c_r, c_b], 2)
        is_in_centers = center_deltas.min(dim=-1).values > 0.0
        anchor_filter = is_in_centers.sum(dim=0) > 0
        geometry_relation = is_in_centers[:, anchor_filter]

        return anchor_filter, geometry_relation

    def simota_matching(self, cost, pair_wise_ious, gt_classes, num_gt, fg_mask):
        matching_matrix = torch.zeros_like(cost, dtype=torch.uint8)

        n_candidate_k = min(10, pair_wise_ious.size(1))
        topk_ious, _ = torch.topk(pair_wise_ious, n_candidate_k, dim=1)
        dynamic_ks = torch.clamp(topk_ious.sum(1).int(), min=1)
        for gt_idx in range(num_gt):
            _, pos_idx = torch.topk(
                cost[gt_idx], k=dynamic_ks[gt_idx], largest=False
            )
            matching_matrix[gt_idx][pos_idx] = 1

        del topk_ious, dynamic_ks, pos_idx

        anchor_matching_gt = matching_matrix.sum(0)
        # deal with the case that one anchor matches multiple ground-truths
        if anchor_matching_gt.max() > 1:
            multiple_match_mask = anchor_matching_gt > 1
            _, cost_argmin = torch.min(cost[:, multiple_match_mask], dim=0)
            matching_matrix[:, multiple_match_mask] *= 0
            matching_matrix[cost_argmin, multiple_match_mask] = 1
        fg_mask_inboxes = anchor_matching_gt > 0
        num_fg = fg_mask_inboxes.sum().item()

        fg_mask[fg_mask.clone()] = fg_mask_inboxes

        matched_gt_inds = matching_matrix[:, fg_mask_inboxes].argmax(0)
        gt_matched_classes = gt_classes[matched_gt_inds]

        pred_ious_this_matching = (matching_matrix * pair_wise_ious).sum(0)[
            fg_mask_inboxes
        ]
        return num_fg, gt_matched_classes, pred_ious_this_matching, matched_gt_inds

    # def visualize_assign_result(self, xin, labels=None, imgs=None, save_prefix="assign_vis_"):
    #     # original forward logic
    #     outputs, x_shifts, y_shifts, expanded_strides = [], [], [], []
    #     # TODO: use forward logic here.

    #     for k, (cls_conv, reg_conv, stride_this_level, x) in enumerate(
    #         zip(self.cls_convs, self.reg_convs, self.strides, xin)
    #     ):
    #         x = self.stems[k](x)
    #         cls_x = x
    #         reg_x = x

    #         cls_feat = cls_conv(cls_x)
    #         cls_output = self.cls_preds[k](cls_feat)
    #         reg_feat = reg_conv(reg_x)
    #         reg_output = self.reg_preds[k](reg_feat)
    #         obj_output = self.obj_preds[k](reg_feat)

    #         output = torch.cat([reg_output, obj_output, cls_output], 1)
    #         output, grid = self.get_output_and_grid(output, k, stride_this_level, xin[0].type())
    #         x_shifts.append(grid[:, :, 0])
    #         y_shifts.append(grid[:, :, 1])
    #         expanded_strides.append(
    #             torch.full((1, grid.shape[1]), stride_this_level).type_as(xin[0])
    #         )
    #         outputs.append(output)

    #     outputs = torch.cat(outputs, 1)
    #     bbox_preds = outputs[:, :, :4]  # [batch, n_anchors_all, 4]
    #     obj_preds = outputs[:, :, 4:5]  # [batch, n_anchors_all, 1]
    #     cls_preds = outputs[:, :, 5:]  # [batch, n_anchors_all, n_cls]

    #     # calculate targets
    #     total_num_anchors = outputs.shape[1]
    #     x_shifts = torch.cat(x_shifts, 1)  # [1, n_anchors_all]
    #     y_shifts = torch.cat(y_shifts, 1)  # [1, n_anchors_all]
    #     expanded_strides = torch.cat(expanded_strides, 1)

    #     nlabel = (labels.sum(dim=2) > 0).sum(dim=1)  # number of objects
    #     for batch_idx, (img, num_gt, label) in enumerate(zip(imgs, nlabel, labels)):
    #         img = imgs[batch_idx].permute(1, 2, 0).to(torch.uint8)
    #         num_gt = int(num_gt)
    #         if num_gt == 0:
    #             fg_mask = outputs.new_zeros(total_num_anchors).bool()
    #         else:
    #             gt_bboxes_per_image = label[:num_gt, 1:5]
    #             gt_classes = label[:num_gt, 0]
    #             bboxes_preds_per_image = bbox_preds[batch_idx]
    #             _, fg_mask, _, matched_gt_inds, _ = self.get_assignments(  # noqa
    #                 batch_idx, num_gt, gt_bboxes_per_image, gt_classes,
    #                 bboxes_preds_per_image, expanded_strides, x_shifts,
    #                 y_shifts, cls_preds, obj_preds,
    #             )

    #         img = img.cpu().numpy().copy()  # copy is crucial here
    #         coords = torch.stack([
    #             ((x_shifts + 0.5) * expanded_strides).flatten()[fg_mask],
    #             ((y_shifts + 0.5) * expanded_strides).flatten()[fg_mask],
    #         ], 1)

    #         xyxy_boxes = cxcywh2xyxy(gt_bboxes_per_image)
    #         save_name = save_prefix + str(batch_idx) + ".png"
    #         img = visualize_assign(img, xyxy_boxes, coords, matched_gt_inds, save_name)
    #         logger.info(f"save img to {save_name}")

            
            
            
            
          # self.hw = [x.shape[-2:] for x in outputs]
          # # [batch, n_anchors_all, 85]
          # outputs = torch.concat([torch.flatten(x, start_dim=2) for x in outputs], dim=2)
          # outputs = torch.tensor(outputs.detach().cpu().numpy().transpose(0,2,1),dtype=torch.float32).to(self.device)
          # if inference_mode:
          #   return self.decode_outputs(outputs.detach().cpu().numpy())
          # else:
          #   return outputs
         
    
if __name__ == "__main__":
    model = IR_RGB_Model(num_classes=3,anchors=1)
    input_rgb = torch.rand((1,3,640,640))
    input_ir = torch.rand((1,3,640,640))
    outs = model(input_rgb,input_ir)
    # output = non_max_suppression(outs)
    # print(output[0].shape)
        
        
        