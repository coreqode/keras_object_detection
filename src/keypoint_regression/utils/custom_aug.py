import os 
import cv2
import numpy as np 
from dataset import Uplara
import matplotlib.pyplot as plt
import random
from PIL import Image
import csv

class RandomRotate(object):
    def __init__(self, image, angle , pts):
        self.image = image
        self.angle = angle
        self.pts = np.array(pts)


    def rotate_im(self):
        (h, w) = self.image.shape[:2]
        (cX, cY) = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D((cX, cY), self.angle, 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))
        self.reso_factor = nW / w
        M[0, 2] += (nW / 2) - cX
        M[1, 2] += (nH / 2) - cY
        image = cv2.warpAffine(self.image, M, (nW, nH))
        image = cv2.resize(image, (w,h))
        return image

    def rotate_box(self):
        (h, w) = self.image.shape[:2]
        (cX, cY) = (w // 2, h // 2)
        pts = self.pts.reshape(-1,2)
        pts = np.hstack((pts, np.ones((pts.shape[0],1), dtype = type(pts[0][0]))))
        M = cv2.getRotationMatrix2D((cX, cY), self.angle, 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))
        # adjust the rotation matrix to take into account translation
        M[0, 2] += (nW / 2) - cX
        M[1, 2] += (nH / 2) - cY
        # Prepare the vector to be transformed
        calculated = np.dot(M,pts.T).T
        calculated = calculated.reshape(-1,50)
        calculated = calculated / self.reso_factor
        return calculated

if __name__ == "__main__":
    dataset = Uplara("/home/noldsoul/Desktop/Uplara/keras_object_detection/src/keypoint_regression/keypoint_dataset.csv","/home/noldsoul/Desktop/Uplara/keras_object_detection/src/keypoint_regression/keypoint_dataset/", transform = None, augmentation=True)
    angles = range(0, 360, 20)
    path = "/home/noldsoul/Desktop/Uplara/keras_object_detection/src/keypoint_regression/aug_keypoints_dataset/"
    with open('augmented_keypoint_dataset.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        top_row = ['foot_id', 'url']
        lists = []
        for i in range(53):
            lists.append(i)
        lists[-1] = 'angle'
        top_row = top_row + lists
        writer.writerow(top_row)
        count = 0
        for angle in angles:
            for idx in range(len(dataset)):
                image, pts, foot_id = dataset[idx]
                aug = RandomRotate(image, angle, pts)
                image = aug.rotate_im()
                pts = aug.rotate_box()
                image = Image.fromarray(image)
                foot_id = str(foot_id) + "_roa"
                image.save(path + foot_id +"_"+str(angle)+".jpg")
                lists = []
                lists = [foot_id,'url', *pts[0] ,angle]
                writer.writerow(lists )
                print('processing: ', count)
                count+=1    


