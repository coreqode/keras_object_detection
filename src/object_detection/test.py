import sys
import numpy as np
import cv2
import glob
import tensorflow as tf
from model.BlazeFace import BlazeFace
from model.MobileNetV2 import MobileNetV2
import argparse
from utils.dataset import *
import pandas as pd
from PIL import Image

class Uplara():
    def __init__(self, dataset_path, image_path, transform = None, image_size = 224, train_from_augmentation = False, augmentation = False): ##Augmentation enables the augmentation.py to use Dataset
        self.dataset = pd.read_csv(dataset_path)
        self.foot_id = self.dataset['foot_id']
        # self.image_url = self.dataset['url']
        self.transform  = transform
        self.image_size = image_size
        self.augmentation = augmentation
        self.train_from_augmentation = train_from_augmentation
        self.image_path = image_path
    def __getitem__(self, idx):
        # print(self.foot_id[idx])
        # image = Image.open(urllib.request.urlopen(self.image_url[idx]))  #when scrapping images directly
        image_path = self.image_path + str(self.foot_id[idx]) + ".jpg"  #Using already scrapped images
        image = Image.open(image_path)
        image = image.resize((self.image_size, self.image_size))
        image = image.convert('RGB')
        image = np.array(image)

        if not self.train_from_augmentation:
            ###################For Left Foot#######################
            l_xmin = np.min(self.dataset.loc[idx][::2][1:26])
            l_ymin = np.max(self.dataset.loc[idx][1:][::2][1:26])
            l_xmax = np.max(self.dataset.loc[idx][::2][1:26])
            l_ymax = np.min(self.dataset.loc[idx][1:][::2][1:26])
            #transformation of coordinates
            l_xmin = int((l_xmin+0.5) * self.image_size)
            l_ymin = int((l_ymin-0.5) * -self.image_size)
            l_xmax = int((l_xmax+0.5) * self.image_size)
            l_ymax = int((l_ymax-0.5) * -self.image_size)
            l_prob = self.dataset.loc[idx][-2]
            ###################For Right Foot #######################
            r_xmin = np.min(self.dataset.loc[idx][52:][::2][0:25])
            r_ymin = np.max(self.dataset.loc[idx][51:][::2][1:26])
            r_xmax = np.max(self.dataset.loc[idx][52:][::2][0:25])
            r_ymax = np.min(self.dataset.loc[idx][51:][::2][1:26])
            #transformation of coordinates
            r_xmin = int((r_xmin+0.5) * self.image_size)
            r_ymin = int((r_ymin-0.5) * -self.image_size)
            r_xmax = int((r_xmax+0.5) * self.image_size)
            r_ymax = int((r_ymax-0.5) * -self.image_size)
            r_prob =self.dataset.loc[idx][-1]
        else:
            ### for left bbox###
            l_xmin = self.dataset['l_xmin'][idx]
            l_ymin = self.dataset['l_ymin'][idx]
            l_xmax = self.dataset['l_xmax'][idx]
            l_ymax = self.dataset['l_ymax'][idx]
            l_prob = self.dataset['l_label'][idx]

            #### for right bbox ####
            r_xmin = self.dataset['r_xmin'][idx]
            r_ymin = self.dataset['r_ymin'][idx]
            r_xmax = self.dataset['r_xmax'][idx]
            r_ymax = self.dataset['r_ymax'][idx]
            r_prob = self.dataset['r_label'][idx]

            if not (self.image_size == 512):
                ratio = self.image_size / 512
                l_xmin, l_ymin, l_xmax, l_ymax = l_xmin*ratio , l_ymin*ratio , l_xmax*ratio ,l_ymax*ratio 
                r_xmin, r_ymin, r_xmax, r_ymax =  r_xmin * ratio, r_ymin * ratio, r_xmax * ratio, r_ymax * ratio

        l_box = [l_xmin, l_ymin, l_xmax, l_ymax]
        r_box = [r_xmin, r_ymin, r_xmax, r_ymax]
        boxes = [l_box, r_box]
        labels = [l_prob, r_prob]
        target = self.encoder(boxes, labels)
        return image, boxes, labels, target,  self.foot_id[idx]

    def encoder(self, boxes, labels):
        target = np.zeros(( 10))
        # for left box
        if not (labels[0] == 0):
            xmin, ymin, xmax, ymax = boxes[0]
            width = xmax - xmin
            height = ymax - ymin
            center_x = (xmin + xmax) / 2
            center_y = (ymin + ymax) / 2
            label = labels[0]
            target[0] = label
            target[2:6] = center_x, center_y, width, height
    # for right box
        if not (labels[1] == 0):
            xmin, ymin, xmax, ymax = boxes[1]
            width = xmax - xmin
            height = ymax - ymin
            center_x = (xmin + xmax) / 2
            center_y = (ymin + ymax) / 2
            label = labels[1]
            target[1] = label
            target[6:10] = center_x, center_y, width, height
        return target

    def __len__(self):
        return len(self.foot_id)


def worst_cases(reg_losses):
        print(reg_losses)
        sorted_reg_losses = sorted(reg_losses, reverse = True)
        top_200 = sorted_reg_losses[:200]
        print(sorted_reg_losses)
        top_200_indices = []
        for i in top_200:
            top_200_indices.append(reg_losses.index(i))
        print(top_200_indices)
        index = 0
        for i in top_200_indices:
            print(i)
            image, boxes, labels, target, foot_id  = dataset[i]
            orig_frame = image
            image = image.astype('float32')
            image = (image / 127.5 ) - 1
            image = np.expand_dims(image, axis = 0)
            output = model.predict(image)
            l_cx, l_cy = output[:,2], output[:,3]
            l_width, l_height = output[:,4],output[:,5]
            l_xmin, l_ymin = l_cx - (l_width/2), l_cy - (l_height/2)
            l_xmax ,l_ymax = l_cx + (l_width/2), l_cy + (l_height/2) 
            r_cx, r_cy = output[:,6], output[:,7]
            r_width, r_height= output[:,8],  output[:,9]
            r_xmin, r_ymin = r_cx - (r_width/2),r_cy - (r_height/2)
            r_xmax,r_ymax = r_cx + (r_width/2),r_cy + (r_height/2)
            l_prob, r_prob = output[:,0],output[:,1]

            if l_prob > 0.5:
                cv2.rectangle(orig_frame,(l_xmin, l_ymin), (l_xmax, l_ymax), (0,255,0), 2 )
                cv2.putText(orig_frame, 'Left', (l_xmin, l_ymin-5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,255,0), 1)
            if r_prob > 0.5:
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
            image = Image.fromarray(orig_frame)
            image.save("/home/noldsoul/Desktop/Uplara/keras_object_detection/src/object_detection/worst_cases/"+ str(index) + "_worst_cases" +".jpg")
            index +=1


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    # hyperparameters
    args.add_argument('--input_shape', type=int, default=224)
    args.add_argument('--train_batch_size', type=int, default=1)
    args.add_argument('--val_batch_size', type=int, default=1)
    args.add_argument('--epochs', type=int, default= 10)
    args.add_argument('--learning_rate', type=int, default=0.001)
    args.add_argument('--num_data', type=int, default=423)
    args.add_argument('--shuffle_buffer', type=int, default=2048)
    args.add_argument('--train', type=bool, default=True)
    args.add_argument('--checkpoint_path', type=str, default="gs://chandradeep_data/best_model.ckpt")
    args.add_argument('--dataset_path', type=str, default="/home/noldsoul/Desktop/Uplara/dataset/newest.csv")
    args.add_argument('--image_dir', type=str, default="/home/noldsoul/Desktop/Uplara/dataset/uplara_images/")
    args.add_argument('--trainset_path', type=str, default="/home/noldsoul/Desktop/Uplara/keras_object_detection/src/object_detection/utils/trainset_300k.record")
    args.add_argument('--valset_path', type=str, default="/home/noldsoul/Desktop/Uplara/keras_object_detection/src/object_detection/utils/valset_300k.record")
    config = args.parse_args()

    with tf.device("/CPU"):
        dataset = Uplara(config.dataset_path, config.image_dir)
        # model = MobileNetV2(config).build_model()
        model = BlazeFace(config).build_model()
        model.load_weights("/home/noldsoul/Downloads/blazenet_model_300k_0.32.hdf5")
        model.summary()
        count = 0
        reg_losses = []
        for i in range(2000):
            image, boxes, labels, target, foot_id  = dataset[i]
            orig_frame = image
            image = image.astype('float32')
            image = (image / 127.5 ) - 1
            image = np.expand_dims(image, axis = 0)
            output = model.predict(image)
            l_cx, l_cy = output[:,2], output[:,3]
            l_width, l_height = output[:,4],output[:,5]
            l_xmin, l_ymin = l_cx - (l_width/2), l_cy - (l_height/2)
            l_xmax ,l_ymax = l_cx + (l_width/2), l_cy + (l_height/2) 
            r_cx, r_cy = output[:,6], output[:,7]
            r_width, r_height= output[:,8],  output[:,9]
            r_xmin, r_ymin = r_cx - (r_width/2),r_cy - (r_height/2)
            r_xmax,r_ymax = r_cx + (r_width/2),r_cy + (r_height/2)
            l_prob, r_prob = output[:,0],output[:,1]

            l_reg_loss  = np.abs(l_cx - target[2]) + np.abs(l_cy - target[3]) + np.abs(l_width - target[4]) + np.abs(l_height - target[5])
            r_reg_loss = np.abs(r_cx - target[6]) + np.abs(r_cy - target[7]) + np.abs(r_width - target[8]) + np.abs(r_height - target[9])
            reg_loss = l_reg_loss + r_reg_loss
            reg_losses.append(reg_loss)
        
        # worst_cases(reg_losses)
    


            if l_prob > 0.5:
                cv2.rectangle(orig_frame,(l_xmin, l_ymin), (l_xmax, l_ymax), (0,255,0), 2 )
                cv2.putText(orig_frame, 'Left', (l_xmin, l_ymin-5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,255,0), 1)
            if r_prob > 0.5:
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
            print(foot_id)
            image.save("/home/noldsoul/Desktop/Uplara/keras_object_detection/src/object_detection/results/"+ str(foot_id) +".jpg")
            if count > 200:
                break
            # count +=1
                # plt.imshow(image.astype(np.uint8))
                # plt.show()
                # image = (image[0] / 255 ) - 1
                # plt.imshow(image)
                # plt.show(image)
