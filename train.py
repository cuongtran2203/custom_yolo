import torch
import cv2
import numpy as np
from Model_IR_RGB_YOLO import IR_RGB_Model
from dataloader import YoloDataset,collate_fn
from process_output_input import *
from process_coupled_head import *
from torch.utils.data import DataLoader  
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()
from torchmetrics.detection import MeanAveragePrecision
import time
from checkpoints import *
def evaluate(model, dataloader):
    model = IR_RGB_Model(num_classes=2,training=False,decode_in_inference=True)

    progress_bar = tqdm(total=len(dataloader), dynamic_ncols=True, position=0, leave=False, desc='Validation')

    model.eval()
    metric = MeanAveragePrecision(iou_type="bbox")

    time_start = time.time()

    for i, (rgb, ir, targets) in enumerate(dataloader):

        with torch.inference_mode():
            outputs = model(rgb, ir, targets)
        detections = postprocess(outputs,num_classes=2,conf_thre=0.25)
        metric.update(detections, targets)
        progress_bar.update()

    total_time = time.time() - time_start
    avg_inf_time = total_time / ( len(dataloader) * BATCH_SIZE )

    metric_vals = metric.compute()

    return metric_vals, avg_inf_time

if __name__ == "__main__":
    # Định nghĩa hyper params
    EPOCHS = 50
    LR = 0.0001
    BATCH_SIZE = 8
    training = True
    # optimizer selected by the original work
    if training:
        model = IR_RGB_Model(num_classes=2,training=training)
    else:
        model = IR_RGB_Model(num_classes=2,training=False,decode_in_inference=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)
    #Khởi tạo dataloader
    
    img_folder = 'RGB_IR_dataset/train/images'  
    label_folder = 'RGB_IR_dataset/train/labels'  
    img_folder_valid = 'RGB_IR_dataset/val/images'
    label_folder_valid = 'RGB_IR_dataset/val/labels' 
    dataset_train = YoloDataset(img_folder=img_folder)  
    dataset_val = YoloDataset(img_folder=img_folder_valid)  
    data_loader_train = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=0,collate_fn=collate_fn)
    data_loader_valid = DataLoader(dataset_val, batch_size=BATCH_SIZE, shuffle=True, num_workers=0,collate_fn=collate_fn)
    
    
    for e in EPOCHS:
        for data in tqdm(data_loader_train):
            rgb,ir,labels = data
            
            loss,iou_loss,conf_loss,l1_loss,num_fg = model(rgb,ir,labels,rgb)
            
            loss.backward()
            optimizer.step()
            
            
            # Logging training
            writer.add_scalar("Total Loss/Train",loss,e)
            writer.add_scalar("Iou_loss/Train",iou_loss,e)
            writer.add_scalar("Conf_loss/Train",conf_loss,e)
            writer.add_scalar("L1_loss/Train",l1_loss,e)
            writer.add_scalar("num_fg/Train",num_fg,e)
        
            
        evaluate(model,data_loader_valid)
        save_checkpoint(e,False,"output","model_ir_rgb")  
            
            
            
            
            
            
            
            
            
            
            
    
    
