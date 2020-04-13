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
model.load_state_dict(torch.load('/home/noldsoul/Desktop/mobilenet_objectdetection_SGD.pt', map_location = torch.device(device)))
model.eval()
model.to(device)
# video_path = '/home/deeplearning/Desktop/chandradeep/Segmentation/test_videos/Video 3.avi'
count = 0
# for path in glob.glob("/home/noldsoul/Desktop/Uplara/dataset/uplara_images/"+"*.jpg"):
transform = torchvision.transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.2, 0.2, 0.2))])
for idx in range(200):
    dataset = Uplara("/home/noldsoul/Desktop/Uplara/dataset/newest.csv", image_size = 224, augmentation = True)
    _, boxes, labels, foot_id = dataset[idx]

    image_path = "/home/noldsoul/Desktop/Uplara/dataset/uplara_images/"+str(foot_id)+".jpg"
    image = Image.open(image_path)
    image = image.convert('RGB')
    image = np.array(image)
    frame = cv2.resize(image, (224, 224)) 
    orig_frame = frame
    t0 = time.time()
    frame = transform(frame)
    frame = frame.unsqueeze(0)
    frame = frame.to(device)
    output = model(frame)
    l_cx, l_cy = output[:,2], output[:,3]
    l_width, l_height = output[:,4],output[:,5]
    l_xmin, l_ymin = l_cx - (l_width/2), l_cy - (l_height/2)
    l_xmax ,l_ymax = l_cx + (l_width/2), l_cy + (l_height/2) 
    r_cx, r_cy = output[:,6], output[:,7]
    r_width, r_height= output[:,8],  output[:,9]
    r_xmin, r_ymin = r_cx - (r_width/2),r_cy - (r_height/2)
    r_xmax,r_ymax = r_cx + (r_width/2),r_cy + (r_height/2)
    l_prob, r_prob = output[:,0],output[:,1]

    if l_prob > 0.6:
        cv2.rectangle(orig_frame,(l_xmin, l_ymin), (l_xmax, l_ymax), (0,255,0), 2 )
        cv2.putText(orig_frame, 'Left', (l_xmin, l_ymin-5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,255,0), 1)
    if r_prob > 0.6:
        cv2.rectangle(orig_frame,(r_xmin, r_ymin), (r_xmax, r_ymax), (0,255,0), 2 )
        cv2.putText(orig_frame, 'Right', (r_xmin, r_ymin-5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,255,0), 1)

    g_l_xmin, g_l_ymin, g_l_xmax, g_l_ymax = boxes[0][0], boxes[0][1], boxes[0][2], boxes[0][3]
    g_r_xmin, g_r_ymin, g_r_xmax, g_r_ymax = boxes[1][0], boxes[1][1], boxes[1][2], boxes[1][3]
    
    if not (labels[0]==0):
        cv2.rectangle(orig_frame,(g_l_xmin, g_l_ymin), (g_l_xmax, g_l_ymax), (255,0,0), 2 )
        cv2.putText(orig_frame, 'Left', (g_l_xmin, g_l_ymin-5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,0,0), 1)
    if not (labels[1]==0):
        cv2.rectangle(orig_frame,(g_r_xmin, g_r_ymin), (g_r_xmax, g_r_ymax), (255,0,0), 2 )
        cv2.putText(orig_frame, 'Right', (g_r_xmin, g_r_ymin-5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,0,0), 1)

    # plt.imshow(orig_frame)
    # plt.show()
    image = Image.fromarray(orig_frame)
    image.save("/home/noldsoul/Desktop/Uplara/MobileNet_ObjectDetection/src/phase1/results/"+ str(foot_id) +".jpg")
    if count > 200:
        break
    count +=1

  
    # print("FPS = ", 1/(time.time()-t0))
    # print("##############End################")