import os
import time
import torch
import matplotlib.pyplot as plt
import cv2
import numpy as np
import seaborn as sns
import torchvision
from model.Net import MobilenetV2_fully_connected
import glob
from PIL import Image
from utils.dataset import Uplara
from torchvision import transforms

device = ('cuda' if torch.cuda.is_available() else 'cpu')

model = MobilenetV2_fully_connected(pretrained=False)
model.load_state_dict(torch.load('/home/noldsoul/Desktop/Uplara/weights/keypoint/keypoint_regression_36.pt', map_location = torch.device(device)))
model.eval()
model.to(device)
# video_path = '/home/deeplearning/Desktop/chandradeep/Segmentation/test_videos/Video 3.avi'
count = 0
# for path in glob.glob("/home/noldsoul/Desktop/Uplara/dataset/uplara_images/"+"*.jpg"):
transform = torchvision.transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.2, 0.2, 0.2))])
for idx in range(200):
    dataset = Uplara("/home/noldsoul/Desktop/Uplara/dataset/keypoint/keypoint_dataset.csv", "/home/noldsoul/Desktop/Uplara/dataset/keypoint/keypoint_dataset/",image_size = 224, augmentation = True)
    image,  all_pts, foot_id = dataset[idx]
    frame = cv2.resize(image, (224, 224)) 
    frame = transform(image)
    frame = frame.unsqueeze(0)
    frame = frame.to(device)
    output = model(frame)
    pred_pts = output.cpu().detach().numpy()[0]
    pred_x, pred_y = pred_pts[::2], pred_pts[1:][::2]
    for x,y in zip(pred_x, pred_y):
        cv2.circle(image, (x,y), 1, (0,255,0), 2)
    
    gt_x, gt_y = all_pts[::2], all_pts[1:][::2]
    for x,y in zip(gt_x, gt_y):
        cv2.circle(image, (x,y),1, (255,0,0), 2)
    # plt.imshow(image)
    # plt.show()
    
    image = Image.fromarray(image)
    image.save("/home/noldsoul/Desktop/Uplara/MobileNet_ObjectDetection/src/phase2/results/"+ str(foot_id) +".jpg")
    if count > 200:
        break
    count +=1

  
    # print("FPS = ", 1/(time.time()-t0))
    # print("##############End################")