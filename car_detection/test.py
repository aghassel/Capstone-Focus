import cv2
from torch.utils.data import DataLoader, Dataset
import os
import torch
from PIL import Image
from torchvision import transforms
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm


# Create the dataset
img_path = 'D:/Kitti8-001/Kitti8/test/image'
bbox_path = 'D:/Kitti8-001/Kitti8/test/label'

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

def calculate_iou(pred, gt):
    x1 = max(pred[0], gt[0])
    y1 = max(pred[1], gt[1])
    x2 = min(pred[2], gt[2])
    y2 = min(pred[3], gt[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    pred_area = (pred[2] - pred[0]) * (pred[3] - pred[1])
    gt_area = (gt[2] - gt[0]) * (gt[3] - gt[1])

    union = pred_area + gt_area - intersection

    iou = intersection / union
    return iou

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

display = False

IoUs = []
# Iterate over the dataset, gather mean IoU
for img_name in tqdm(os.listdir(img_path)):
    cv2_img = cv2.imread(os.path.join(img_path, img_name))
    img = cv2.resize(cv2_img, (1280, 720))
    pred = model(img)

    pred = pred.xyxy[0].cpu().detach().numpy()
    pred = [x for x in pred if x[5] == 2]
    pred = [(x[0], x[1], x[2], x[3]) for x in pred]

    with open(os.path.join(bbox_path, img_name.replace('.png', '.txt')), 'r') as f:
        gt = f.readline().split(' ')
        gt = [float(x) for x in gt if x.replace('.','',1).isdigit()]
        gt = (gt[4], gt[5], gt[6], gt[7])
        
    # scale the bounding box coordinates back to the original frame size
    gt = (gt[0]*img.shape[1], gt[1]*img.shape[0], gt[2]*img.shape[1], gt[3]*img.shape[0])

    
    # display the image with the bounding box
    if display:
        for i in pred:
            cv2.rectangle(img, (int(i[0]), int(i[1])), (int(i[2]), int(i[3])), (0, 255, 0), 2)
            cv2.putText(img, 'Car', (int(i[0]), int(i[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

        cv2.imshow('Frame', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    if len(pred) > 0:
        for i in pred:
            IoUs.append(calculate_iou(i, gt))

print(f'Mean IoU over the dataset: {np.mean(IoUs)}')


