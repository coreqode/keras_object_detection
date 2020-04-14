import tensorflow as tf
import numpy as np
import cv2
import pickle
import glob
import os
import time
import argparse
from model.Net import BlazeFace
from utils.loss import custom_loss
from utils.dataset import dataloader
from utils.read_tfrecord import create_dataset


if __name__ == "__main__":

    args = argparse.ArgumentParser()
    # hyperparameters
    args.add_argument('--input_shape', type=int, default=224)
    args.add_argument('--train_batch_size', type=int, default=8)
    args.add_argument('--val_batch_size', type=int, default=8)
    args.add_argument('--epochs', type=int, default=100)
    args.add_argument('--learning_rate', type=int, default=0.01)
    args.add_argument('--num_data', type=int, default=423)
    args.add_argument('--shuffle_buffer', type=int, default=1024)
    args.add_argument('--train', type=bool, default=True)
    args.add_argument('--checkpoint_path', type=str, default="./checkpoint/best_model.ckpt")
    # args.add_argument('--dataset_path', type=str, default="/home/chan/data/augmented_dataset_300k.csv")
    # args.add_argument('--image_dir', type=str, default="/home/chan/data/augmented_images_300k/")
    args.add_argument('--trainset_path', type=str, default="/home/noldsoul/Desktop/Uplara/keras_object_detection/src/object_detection/utils/trainset_300k.record")
    args.add_argument('--valset_path', type=str, default="/home/noldsoul/Desktop/Uplara/keras_object_detection/src/object_detection/utils/valset_300k.record")

    config = args.parse_args()

    # tf.keras.backend.clear_session()
    # TPU_WORKER = 'grpc://' + '10.108.85.226:8470'
    # # tf.config.experimental_connect_to_cluster(TPU_WORKER)
    # resolver = tf.distribute.cluster_resolver.TPUClusterResolver(TPU_WORKER)
    # tf.config.experimental_connect_to_cluster(resolver)
    # tf.tpu.experimental.initialize_tpu_system(resolver)
    # strategy = tf.distribute.experimental.TPUStrategy(resolver)
    # tf.compat.v1.disable_eager_execution() 

    # tpu_model = tf.contrib.tpu.keras_to_tpu_model( model,strategy=tf.contrib.tpu.TPUDistributionStrategy(tf.contrib.cluster_resolver.TPUClusterResolver(TPU_WORKER)))
    strategy = tf.distribute.get_strategy()
    with strategy.scope():
        model = BlazeFace(config).build_model()
        opt = tf.keras.optimizers.Adam(learning_rate = config.learning_rate)
        model.compile(loss= custom_loss, optimizer=opt)
        print(model.summary())

    train_gen = create_dataset(config.train_batch_size, config.trainset_path, config.shuffle_buffer)
    STEP_SIZE_TRAIN = int((0.95 * config.num_data) // config.train_batch_size)
    val_gen = create_dataset(config.val_batch_size, config.valset_path, config.shuffle_buffer)
    STEP_SIZE_VAL = int((0.05 * config.num_data) // config.val_batch_size)

    history = model.fit(x=train_gen ,epochs = config.epochs, 
                                    steps_per_epoch = STEP_SIZE_TRAIN,
                                    validation_data = val_gen,
                                    validation_steps = STEP_SIZE_VAL,
                                    # callbacks = [early_stopping, checkpoint],
                                    verbose=1,
                                    shuffle=True,
                                    use_multiprocessing=False)

    # train(model, config)
