import sys
import numpy as np
import cv2
import glob
import tensorflow as tf
from model.Net import BlazeFace
import argparse
from utils.dataset import *



if __name__ == "__main__":
    args = argparse.ArgumentParser()
    # hyperparameters
    args.add_argument('--input_shape', type=int, default=224)
    args.add_argument('--train_batch_size', type=int, default=4)
    args.add_argument('--val_batch_size', type=int, default=4)
    args.add_argument('--epochs', type=int, default= 10)
    args.add_argument('--learning_rate', type=int, default=0.001)
    args.add_argument('--num_data', type=int, default=423)
    args.add_argument('--shuffle_buffer', type=int, default=2048)
    args.add_argument('--train', type=bool, default=True)
    args.add_argument('--checkpoint_path', type=str, default="gs://chandradeep_data/best_model.ckpt")
    args.add_argument('--dataset_path', type=str, default="/home/noldsoul/Desktop/Uplara/MobileNet_ObjectDetection/src/phase1/utils/augmented_dataset.csv")
    args.add_argument('--image_dir', type=str, default="/home/noldsoul/Desktop/Uplara/MobileNet_ObjectDetection/src/phase1/utils/augmented_images/")
    args.add_argument('--trainset_path', type=str, default="/home/noldsoul/Desktop/Uplara/keras_object_detection/src/object_detection/utils/trainset_300k.record")
    args.add_argument('--valset_path', type=str, default="/home/noldsoul/Desktop/Uplara/keras_object_detection/src/object_detection/utils/valset_300k.record")
    config = args.parse_args()

    model = BlazeFace(config).build_model()
    model.load_weights("/home/noldsoul/Desktop/Uplara/keras_object_detection/src/object_detection/model_300k.h5")
    model.summary()
    dataset = dataloader(config)
    # dataset = Uplara(config)
    image, label  =next(iter(dataset))
    output = model(image)
    print(output[0])
    # plt.imshow(image.astype(np.uint8))
    # plt.show()
    # image = (image[0] / 255 ) - 1
    # plt.imshow(image)
    # plt.show(image)
