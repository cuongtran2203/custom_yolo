from models.csp_darknet import YOLOv5CSPDarknet


import torch


model = YOLOv5CSPDarknet()
model.eval()
inputs = torch.rand(1, 3, 416, 416)
level_outputs = model(inputs)
for level_out in level_outputs:
    print(tuple(level_out.shape))
