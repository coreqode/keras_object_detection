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

def train(model, config):
    opt = tf.train.AdamOptimizer(learning_rate = 0.01)
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
    
    train_gen = dataloader(config)
    STEP_SIZE_TRAIN = int(0.95 * config.num_data) // config.train_batch_size
    val_gen = dataloader(config, train_generator=False)
    STEP_SIZE_VAL = (0.5 * config.num_data) // config.val_batch_size

    for epoch in range(config.epochs):
        t1 = time.time()
        res = model.fit_generator(generator=train_gen,
                                    steps_per_epoch = STEP_SIZE_TRAIN,
                                    validation_data = val_gen,
                                    validation_steps = STEP_SIZE_VAL,
                                    initial_epoch=epoch,
                                    epochs=epoch + 1,
                                    callbacks = [early_stopping, checkpoint],
                                    verbose=1,
                                    shuffle=True,
                                    use_multiprocessing=False)

if __name__ == "__main__":

    args = argparse.ArgumentParser()
    # hyperparameters
    args.add_argument('--input_shape', type=int, default=224)
    args.add_argument('--train_batch_size', type=int, default=16)
    args.add_argument('--val_batch_size', type=int, default=16)
    args.add_argument('--epochs', type=int, default=100)
    args.add_argument('--num_data', type=int, default=427)
    args.add_argument('--train', type=bool, default=True)
    args.add_argument('--checkpoint_path', type=str, default="./checkpoint/best_model.ckpt")
    args.add_argument('--dataset_path', type=str, default="/home/noldsoul/Desktop/Uplara/MobileNet_ObjectDetection/src/phase1/utils/augmented_dataset.csv")
    args.add_argument('--image_dir', type=str, default="/home/noldsoul/Desktop/Uplara/MobileNet_ObjectDetection/src/phase1/utils/augmented_images/")

    config = args.parse_args()

    # model = BlazeFace(config).build_model()
    
    tf.keras.backend.clear_session()
    TPU_WORKER = 'grpc://' + '10.12.97.242:8470'
    tf.config.experimental_connect_to_host(TPU_WORKER)
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver('grpc://' + '10.12.97.242:8470'])
    tf.tpu.experimental.initialize_tpu_system(resolver)
    strategy = tf.distribute.experimental.TPUStrategy(resolver) 
    with strategy.scope():
        model = BlazeFace(config).build_model()
        train(model, config)