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
from dataset import Uplara
from torchvision import transforms
import csv
import random

class project_points:
    def __init__(self, image, all_pts):
        self.image = image
        self.all_pts = all_pts

    def image_resize(self,  inter = cv2.INTER_AREA):
        dim = None
        frame_h, frame_w = 224,224
        (image_h, image_w) = self.image.shape[:2]
        if image_h > image_w:
            r = frame_h / float(image_h)
            dim = (int(image_w * r), frame_h)
            points = [int(i * r) for i in self.all_pts]
        else:
            r = frame_w / float(image_w)
            dim = (frame_w, int(image_h * r))
            points = [int(i * r) for i in self.all_pts]
        resized = cv2.resize(self.image, dim, interpolation = inter)

        ########### TO fill black padding ###########
        delta_h = frame_h - resized.shape[0]
        delta_w = frame_w - resized.shape[1]
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)
        color = [0,0,0]
        new_im = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value = color)
        x, y = points[::2], points[1:][::2]

        new_points = []
        for x,y in zip(x, y):
            x = x + delta_w // 2
            y = y + delta_h // 2
            # cv2.circle(new_im, (x,y), 1, (0,255,0), 2)
            new_points.append(x)
            new_points.append(y)
        # plt.imshow(new_im)
        # plt.show()
        return new_im, new_points

def get_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, (xB - xA + 1)) * max(0, (yB - yA + 1))
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def write_csv(image, points, foot_id, string):
    path = "/home/noldsoul/Desktop/Uplara/keras_object_detection/src/keypoint_regression/keypoint_dataset/"
    image = Image.fromarray(image)
    foot_id = str(foot_id) + "_" + string 
    image.save(path + foot_id + ".jpg")
    lists = []
    lists = [foot_id, 'url', *points]
    writer.writerow(lists )
    print('processing: ', foot_id)


if __name__ == "__main__":
    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    # model = MobilenetV2_fully_connected(pretrained=False)
    # model.load_state_dict(torch.load('/home/noldsoul/Desktop/Uplara/weights/augmented_3lakh.pt', map_location = torch.device(device)))
    # model.eval()
    # model.to(device)
    # # transform = torchvision.transforms.Compose([transforms.ToTensor(),
    #                                 transforms.Normalize((0.5, 0.5, 0.5), (0.2, 0.2, 0.2))])
    dataset = Uplara("/home/noldsoul/Desktop/Uplara/dataset/newest.csv", "/home/noldsoul/Desktop/Uplara/dataset/uplara_images/", transform = None, get_resized=True)
    
    ## Creating the csv file to write dataset
    with open('keypoint_dataset.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        top_row = ['foot_id', 'url']
        lists = []
        for i in range(50):
            lists.append(i)
        top_row = top_row + lists
        writer.writerow(top_row)
        for i in range(len(dataset)):
            image, all_pts, foot_id, label = dataset[i]
            print(foot_id, i)
            # frame = transform(image).unsqueeze(0)
            # frame = frame.to(device)
            # output = model(frame).cpu().detach().numpy()
            # l_cx, l_cy = output[0,2], output[0,3]
            # l_width, l_height = output[0,4],output[0,5]
            # l_xmin, l_ymin = int(l_cx - (l_width/2)), int(l_cy - (l_height/2))
            # l_xmax ,l_ymax = int(l_cx + (l_width/2)), int(l_cy + (l_height/2)) 
            # r_cx, r_cy = output[0,6], output[0,7]
            # r_width, r_height= output[0,8],  output[0,9]
            # r_xmin, r_ymin = int(r_cx - (r_width/2)) ,int(r_cy - (r_height/2))
            # r_xmax,r_ymax = int(r_cx + (r_width/2)),int(r_cy + (r_height/2))
            # l_prob, r_prob = output[0,0],output[0,1]

            l_gt_points = [np.min(all_pts[0]), np.min(all_pts[1]), np.max(all_pts[0]), np.max(all_pts[1]) ]
            r_gt_points = [np.min(all_pts[2]), np.min(all_pts[3]), np.max(all_pts[2]), np.max(all_pts[3]) ]
            g_l_xmin, g_l_ymin, g_l_xmax, g_l_ymax = l_gt_points[0], l_gt_points[1], l_gt_points[2], l_gt_points[3] 
            g_r_xmin, g_r_ymin, g_r_xmax, g_r_ymax = r_gt_points[0], r_gt_points[1], r_gt_points[2], r_gt_points[3] 

            # l_iou = get_iou(l_gt_points, [l_xmin, l_ymin, l_xmax, l_ymax])
            # r_iou = get_iou(r_gt_points, [r_xmin, r_ymin, r_xmax, r_ymax])
            # print(l_iou, r_iou)
            # cv2.rectangle(image, (l_gt_points[0], l_gt_points[1]), (l_gt_points[2], l_gt_points[3]), (255,0,0), 2)
            # cv2.rectangle(image, (l_xmin, l_ymin), (l_xmax, l_ymax), (0,255,0), 2)
            # plt.imshow(image)
            # plt.show()
            ## check for left image

            if label[0] == 1.0:
                delta_1, delta_2 = g_l_xmin*random.uniform(0,0.15), g_l_ymin*random.uniform(0,0.15)
                delta_3, delta_4 = g_l_xmax*random.uniform(0,0.15), g_l_ymax*random.uniform(0,0.15)
                l_crop_image = image[ max(0,int(g_l_ymin-delta_2)):int(g_l_ymax+delta_4) , max(0,int(g_l_xmin-delta_1)): int(g_l_xmax+delta_3)]
                cropped_l_gt_points = []
                for x, y in zip(all_pts[0],all_pts[1]):
                    cropped_l_gt_points.append(x - g_l_xmin+delta_1)
                    cropped_l_gt_points.append(y - g_l_ymin+delta_2)
                pp = project_points(l_crop_image,cropped_l_gt_points)
                new_image, new_points = pp.image_resize()
                write_csv(new_image, new_points, foot_id, "left")

                ## check for right image
            if label[1] == 1.0:
                delta_1, delta_2 = g_r_xmin*random.uniform(0,0.15), g_r_ymin*random.uniform(0,0.15)
                delta_3, delta_4 = g_r_xmax*random.uniform(0,0.15), g_r_ymax*random.uniform(0,0.15)
                r_crop_image = image[ max(0,int(g_r_ymin-delta_2)):int(g_r_ymax+delta_4),max(0,int(g_r_xmin-delta_1)):int(g_r_xmax+delta_3)]
                cropped_r_gt_points = []
                for x, y in zip(all_pts[2],all_pts[3]):
                        cropped_r_gt_points.append(x - g_r_xmin+delta_1)
                        cropped_r_gt_points.append(y - g_r_ymin+delta_2)
                pp = project_points(r_crop_image,cropped_r_gt_points)
                new_image, new_points = pp.image_resize()
                write_csv(new_image, new_points, foot_id, "right")

