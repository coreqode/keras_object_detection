import tensorflow as tf
import argparse
import matplotlib.pyplot as plt
from pandas import read_csv
import cv2
import glob
import os
import numpy as np
import logging
import pandas as pd 
from PIL import Image
import random

class Uplara():
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
       
        all_pts = self.dataset.loc[idx][2:52]
        target = self.encoder(all_pts)  # To get the encoding of the target
        return image, target, image_path
        
    def encoder(self, all_pts):
        target = np.zeros((50))
        target[0:] = all_pts
        return target

    def __len__(self):
        return len(self.foot_id)

def _float32_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


if __name__ == "__main__":

    args = argparse.ArgumentParser()
    # hyperparameters
    args.add_argument('--input_shape', type=int, default=224)
    args.add_argument('--train_batch_size', type=int, default=128)
    args.add_argument('--val_batch_size', type=int, default=128)
    args.add_argument('--train', type=bool, default=True)
    args.add_argument('--checkpoint_path', type=str, default="./checkpoint/best_model.ckpt")
    args.add_argument('--dataset_path', type=str, default="/home/noldsoul/Desktop/Uplara/keras_object_detection/src/keypoint_regression/keypoint_dataset.csv")
    args.add_argument('--image_dir', type=str, default="/home/noldsoul/Desktop/Uplara/keras_object_detection/src/keypoint_regression/keypoint_images/")

    config = args.parse_args()
    dataset = Uplara(dataset_path = config.dataset_path, image_path = config.image_dir, image_size= config.input_shape)


    TRAIN_FILEPATH =  "./trainset_300k.record"
    VAL_FILEPATH =  "./valset_300k.record"
    trainwriter = tf.io.TFRecordWriter(TRAIN_FILEPATH)
    valwriter = tf.io.TFRecordWriter(VAL_FILEPATH)
    # Define the features of your tfrecord
    indices = np.arange(len(dataset))
    random.shuffle(indices)
    train_indices = indices[:int(0.95 * len(dataset))]
    val_indices = indices[int(0.95 * len(dataset)) : ]

    # for trainset
    count = 0
    for i in train_indices:
        print("processing :", count)
        count+=1
        image, label, image_path = dataset[i]

        feature = {'image':  _bytes_feature(tf.compat.as_bytes(open(image_path, 'rb').read())),
                'points':  _float32_feature(value = label)
                }


        # Serialize to string and write to file
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        trainwriter.write(example.SerializeToString())
    print("trainset completed")
    
    #for valset
    count = 0
    for i in val_indices:
        print("processing :", count)
        count +=1
        image, label, image_path = dataset[i]
        feature = {'image':  _bytes_feature(tf.compat.as_bytes(open(image_path, 'rb').read())),
                'l_prob':  _float32_feature(float(label[0])),
                'r_prob':  _float32_feature(float(label[1])),
                'l_center_x':  _float32_feature(float(label[2])),
                'l_center_y':  _float32_feature(float(label[3])),
                'l_width':  _float32_feature(float(label[4])),
                'l_height':  _float32_feature(float(label[5])),
                'r_center_x':  _float32_feature(float(label[6])),
                'r_center_y':  _float32_feature(float(label[7])),
                'r_width':  _float32_feature(float(label[8])),
                'r_height':  _float32_feature(float(label[9])),
                }


        # Serialize to string and write to file
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        valwriter.write(example.SerializeToString())
    print("valset completed")
