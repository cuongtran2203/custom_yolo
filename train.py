import torch
import cv2
import numpy as np
from Model_IR_RGB_YOLO import IR_RGB_Model
from dataloader import YoloDataset,collate_fn
from process_output_input import *
from process_coupled_head import *
from torch.utils.data import DataLoader  
from tqdm import tqdm
from loguru import Logger
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
    logger = Logger()
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
    
    img_folder = 'RGB_IR_dataset/train'  
    label_folder = 'RGB_IR_dataset/train'  
    img_folder_valid = 'RGB_IR_dataset/val'
    label_folder_valid = 'RGB_IR_dataset/val' 
    dataset_train = YoloDataset(img_folder=img_folder)  
    dataset_val = YoloDataset(img_folder=img_folder_valid)  
    data_loader_train = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=0,collate_fn=collate_fn)
    data_loader_valid = DataLoader(dataset_val, batch_size=BATCH_SIZE, shuffle=True, num_workers=0,collate_fn=collate_fn)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model.to(device)
    for e in range(EPOCHS):
        loss_t = 0
        iou_loss_t = 0
        conf_loss_t = 0
        cls_loss_t = 0
        
        for data in tqdm(data_loader_train):
            rgb,ir,labels = data
            rgb.to(device)
            ir.to(device)
            labels.to(device)
            loss,iou_loss,conf_loss,cls_loss,l1_loss,num_fg = model(rgb,ir,labels,rgb)
            
            loss.backward()
            optimizer.step()
            loss_t += loss.item()
            iou_loss_t += iou_loss.item()
            conf_loss_t += conf_loss.item()
            cls_loss_t += cls_loss.item()
            # Logging training
            writer.add_scalar("Total Loss/Train",loss,e)
            writer.add_scalar("Iou_loss/Train",iou_loss,e)
            writer.add_scalar("Conf_loss/Train",conf_loss,e)
            writer.add_scalar("Cls_loss/Train",cls_loss,e)
            writer.add_scalar("L1_loss/Train",l1_loss,e)
            writer.add_scalar("num_fg/Train",num_fg,e)
            
        loss_t = loss_t/len(data_loader_train)
        iou_loss_t = iou_loss_t/len(data_loader_train)
        conf_loss_t =  conf_loss_t/len(data_loader_train)
        cls_loss_t = cls_loss_t/len(data_loader_train)
        
        
        logger.info(f"Total loss: {loss_t} IOU_Loss: {iou_loss_t} conf_loss: {conf_loss_t} cls_loss: {cls_loss_t} at EPOCH {e}")
        
        metric_vals, avg_inf_time= evaluate(model,data_loader_valid)
        
        mAP = metric_vals["map"]
        mAP_50 =  metric_vals["map_50"]
        mAP_75 =  metric_vals["map_75"]
        writer.add_scalar("mAP/val",mAP,e)
        writer.add_scalar("mAP_50/val",mAP_50,e)
        writer.add_scalar("mAP_75/val",mAP_75,e)
        logger.info(f"EVAL =>>>> mAP: {mAP} mAP 50:{mAP_50} mAP 75: {mAP_75} at epoch {e}")
        save_checkpoint(e,False,"output","model_ir_rgb")  
            
            
            
            
            
            
            
            
            
            
            
    
    
