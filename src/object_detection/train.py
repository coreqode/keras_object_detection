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

def train(model, config):
    opt = tf.keras.optimizers.Adam  (learning_rate = 0.0001)
    model.compile(loss= custom_loss, optimizer=opt)
    # monitor = 'loss'
    # reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    #     monitor=monitor, patience=4)
    early_stopping = tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='min',
                    baseline=None, restore_best_weights=True
                )
    
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
                config.checkpoint_path, monitor='val_loss', verbose=0, save_best_only=False,
                save_weights_only=True, mode='min', save_freq='epoch')
    
    train_gen = create_dataset(config.train_batch_size, config.trainset_path, config.shuffle_buffer)
    STEP_SIZE_TRAIN = int((0.95 * config.num_data) // config.train_batch_size)
    val_gen = create_dataset(config.val_batch_size, config.valset_path, config.shuffle_buffer)
    STEP_SIZE_VAL = int((0.05 * config.num_data) // config.val_batch_size)

    res = model.fit(x=train_gen[0], y = train_gen[1], epochs = config.epochs, 
                                    steps_per_epoch = STEP_SIZE_TRAIN,
                                    validation_data = val_gen,
                                    validation_steps = STEP_SIZE_VAL,
                                    callbacks = [early_stopping, checkpoint],
                                    verbose=1,
                                    shuffle=True,
                                    use_multiprocessing=False)
    print(res.values)

if __name__ == "__main__":

    args = argparse.ArgumentParser()
    # hyperparameters
    args.add_argument('--input_shape', type=int, default=224)
    args.add_argument('--train_batch_size', type=int, default=8)
    args.add_argument('--val_batch_size', type=int, default=8)
    args.add_argument('--epochs', type=int, default=100)
    args.add_argument('--num_data', type=int, default=432)
    args.add_argument('--shuffle_buffer', type=int, default=64)
    args.add_argument('--train', type=bool, default=True)
    args.add_argument('--checkpoint_path', type=str, default="./checkpoint/best_model.ckpt")
    # args.add_argument('--dataset_path', type=str, default="/home/chan/data/augmented_dataset_300k.csv")
    # args.add_argument('--image_dir', type=str, default="/home/chan/data/augmented_images_300k/")
    args.add_argument('--trainset_path', type=str, default="/home/noldsoul/Desktop/Uplara/keras_object_detection/src/object_detection/utils/trainset.record")
    args.add_argument('--valset_path', type=str, default="/home/noldsoul/Desktop/Uplara/keras_object_detection/src/object_detection/utils/valset.record")

    config = args.parse_args()

    # model = BlazeFace(config).build_model()
    
    tf.keras.backend.clear_session()
    # TPU_WORKER = 'grpc://' + '10.12.97.242:8470'
   # tf.config.experimental_connect_to_cluster(TPU_WORKER)
    # resolver = tf.distribute.cluster_resolver.TPUClusterResolver('grpc://' + '10.12.97.242:8470')
    # tf.config.experimental_connect_to_cluster(resolver)
    # tf.tpu.experimental.initialize_tpu_system(resolver)
    # strategy = tf.distribute.experimental.TPUStrategy(resolver)
    # tf.compat.v1.disable_eager_execution() 
    model = BlazeFace(config).build_model()
    # tpu_model = tf.contrib.tpu.keras_to_tpu_model( model,strategy=tf.contrib.tpu.TPUDistributionStrategy(tf.contrib.cluster_resolver.TPUClusterResolver(TPU_WORKER)))
    # with strategy.scope():
    train(model, config)
