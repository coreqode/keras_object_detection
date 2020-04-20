import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import glob
import cv2
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data.dataset import random_split
from PIL import Image
import pandas as pd
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
import urllib.request
import imgaug as ia

## Dataset uses fully connected network as a feature map
class Uplara(Dataset):
    def __init__(self, dataset_path, image_path, transform = None, image_size = 224, training = False, augmentation = False, get_resized = False): ##Augmentation enables the augmentation.py to use Dataset
        self.dataset = pd.read_csv(dataset_path)
        self.foot_id = self.dataset['foot_id']
        self.image_path = image_path
        self.transform  = transform
        self.image_size = image_size
        self.training = training
        self.augmentation = augmentation
        self.get_resized = get_resized
    def __getitem__(self, idx):
        image_path = self.image_path + str(self.foot_id[idx]) + ".jpg"  #Using already scrapped images
        image = Image.open(image_path)
        image = image.resize((self.image_size, self.image_size))
        image = image.convert('RGB')
        image = np.array(image)
       
        if self.get_resized:
            ###################For  Foot#######################
            l_x_pts = self.dataset.loc[idx][::2][1:26]
            l_y_pts = self.dataset.loc[idx][1:][::2][1:26]
            r_x_pts = self.dataset.loc[idx][52:][::2][0:25]
            r_y_pts = self.dataset.loc[idx][51:][::2][1:26]
            #transformation of coordinates
            l_x_pts = [int((pt + 0.5) * self.image_size) for pt in l_x_pts]
            l_y_pts = [int((pt - 0.5) * -self.image_size) for pt in l_y_pts]
            r_x_pts = [int((pt + 0.5) * self.image_size) for pt in r_x_pts]
            r_y_pts = [int((pt - 0.5) * -self.image_size) for pt in r_y_pts]
            #### Probability od foot
            l_prob = self.dataset.loc[idx][-2]
            r_prob =self.dataset.loc[idx][-1]
            # l_xmin, l_ymin, l_xmax, l_ymax = np.min(l_x_pts), np.max(l_y_pts), np.max(l_x_pts), np.min(l_y_pts)
            # cv2.rectangle(image,(l_xmin, l_ymin), (l_xmax, l_ymax), (0,255,0), 2 )
            # plt.imshow(image)
            # plt.show()
            all_pts = [l_x_pts, l_y_pts, r_x_pts, r_y_pts]
            return image, all_pts, self.foot_id[idx], [l_prob, r_prob]   ## For augmenting the data

        ## for doing the audmentation
        if self.augmentation:
                all_pts = self.dataset.loc[idx][2:52]
                return image, all_pts, self.foot_id[idx]

        if self.training: # For training the mode
            all_pts = self.dataset.loc[idx][2:52]

        if self.transform:
            image = self.transform(image)
        else:
            image = torch.from_numpy(image).float()
            image = image.permute([2,0,1])

        target = self.encoder(all_pts)  # To get the encoding of the target
        target = torch.from_numpy(target).float()
        return image, target
        
    def encoder(self, all_pts):
        target = np.zeros((50))
        target[0:] = all_pts
        return target

    def __len__(self):
        return len(self.foot_id)

if __name__ == "__main__":
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.2, 0.2, 0.2))])
    # Dataset = Uplara("/home/noldsoul/Desktop/Uplara/newest.csv", transform = transform)
    dataset = Uplara("/home/noldsoul/Desktop/Uplara/dataset/newest.csv","/home/noldsoul/Desktop/Uplara/dataset/uplara_images/", transform = transform)
    # print(dataset[6])