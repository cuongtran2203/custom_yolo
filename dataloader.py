import glob
import math
import os
import random
import shutil
import time
from itertools import repeat
from multiprocessing.pool import ThreadPool
from pathlib import Path
from threading import Thread

import cv2
import numpy as np
import torch
from PIL import Image, ExifTags
from torch.utils.data import Dataset
from tqdm import tqdm

from utils.general import xyxy2xywh, xywh2xyxy,xyxy2xywhn
from utils.torch_utils import torch_distributed_zero_first
from utils.augmentations import *
from torch.utils.data import DataLoader  


class YoloDataset(Dataset):  
    def __init__(self, img_folder, img_size=[640,640], augment=False):  
        self.img_folder = img_folder  
        self.img_size = img_size  
        self.augment = augment  
        self.images= os.listdir(os.path.join(img_folder,"images","Vis"))
        
          
    def __len__(self):  
        return len(self.images)
    def normalize(self,img):
        img = img / 255.0  # Chuẩn hóa giá trị pixel  
        img = np.transpose(img, (2, 0, 1))  # Chuyển đổi thành (C, H, W) 
        return img
    
    def __getitem__(self, idx):  
        img_path_rgb = os.path.join(self.img_folder,"images","Vis",self.images[idx])
        img_path_ir =  os.path.join(self.img_folder,"images","Ir",self.images[idx])
        label_path = os.path.join(self.img_folder,"labels", self.images[idx].replace('.png', '.txt'))  
          
        # Đọc hình ảnh  
        img_rgb = cv2.imread(img_path_rgb)
        img_ir = cv2.imread(img_path_ir)
        h0, w0 = img_rgb.shape[:2]  # orig hw
        r = self.img_size[0] / max(h0, w0)  # ratio
        if r != 1:  # if sizes are not equal
            interp = cv2.INTER_LINEAR if (self.augment or r > 1) else cv2.INTER_AREA
            img_rgb = cv2.resize(img_rgb, (math.ceil(w0 * r), math.ceil(h0 * r)), interpolation=interp)
            img_ir = cv2.resize(img_ir, (math.ceil(w0 * r), math.ceil(h0 * r)), interpolation=interp)
        h,w = img_rgb.shape[:2]
        # Preprocess image
        img_rgb, ratio, pad = letterbox(img_ir, self.img_size, auto=False, scaleup=False)
        img_ir, ratio, pad = letterbox(img_ir, self.img_size, auto=False, scaleup=False)
        # Normilize ảnh 
        img_rgb = self.normalize(img_rgb)
        img_ir = self.normalize(img_ir)
        shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling

        # Đọc nhãn  
        if os.path.exists(label_path):  
            labels = np.loadtxt(label_path).reshape(-1, 5)  # Đọc nhãn  
        else:  
            labels = np.empty((0, 5))  # Không có nhãn   
        
        if labels.size:  # normalized xywh to pixel xyxy format
            labels[:, 1:] = xywhn2xyxy(labels[:, 1:], ratio[0] * w, ratio[1] * h, padw=pad[0], padh=pad[1])
        nl = len(labels)  # number of labels
        if nl:
            labels[:, 1:5] = xyxy2xywhn(labels[:, 1:5], w=img_rgb.shape[1], h=img_rgb.shape[0], clip=True, eps=1e-3)
  
        return torch.FloatTensor(img_rgb),torch.FloatTensor(img_ir), torch.FloatTensor(labels)
def collate_fn(batch):
    """Batches images, labels, paths, and shapes, assigning unique indices to targets in merged label tensor."""
    im_rgb,im_ir, label = zip(*batch)  # transposed
    for i, lb in enumerate(label):
        lb[:, 0] = i  # add target image index for build_targets()
    return torch.stack(im_rgb, 0), torch.stack(im_ir, 0), torch.cat(label, 0)
# if __name__ == "__main__":
#     img_folder = 'RGB_IR_dataset/train'  
#     label_folder = 'dataset/labels/train'  
#     dataset = YoloDataset(img_folder=img_folder)  
    
#     data_loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0,collate_fn=collate_fn)  
    
#     # Kiểm tra một batch dữ liệu  
#     for imgrgb,imgir, labels in data_loader:  
#         print(imgrgb.shape)  # Kiểm tra kích thước của batch  
#         print(labels.shape)  
#         break  