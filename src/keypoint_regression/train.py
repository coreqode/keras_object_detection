import tensorflow as tf
import numpy as np
import cv2
import pickle
import glob
import os
import time
import argparse
import numpy as np
from model.BlazeFace import BlazeFace
from model.MobileNet import MobileNetV2
from utils.loss import custom_loss
from utils.dataset import dataloader
from utils.read_tfrecord import create_dataset

def scheduler(epoch):
    if epoch < 5:
        return 0.045
    else:
        return 0.045 * tf.math.exp(0.1 * (10 - epoch))

class CustomModelCheckpoint(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        # logs is a dictionary
        if (np.abs(logs['val_loss'] - logs['loss'])) < 0.5:
            self.model.save_weights('./model_300k.hdf5', overwrite=True)

if __name__ == "__main__":

    args = argparse.ArgumentParser()
    # hyperparameters
    args.add_argument('--input_shape', type=int, default=224)
    args.add_argument('--train_batch_size', type=int, default=128 * 8)
    args.add_argument('--val_batch_size', type=int, default=128* 8)
    args.add_argument('--epochs', type=int, default= 150)
    args.add_argument('--learning_rate', type=int, default=0.0000045)
    args.add_argument('--num_data', type=int, default=306342)
    args.add_argument('--shuffle_buffer', type=int, default=2048)
    args.add_argument('--train', type=bool, default=True)
    args.add_argument('--inference', type = bool, default = False) 
    args.add_argument('--checkpoint_path', type=str, default="gs://chandradeep_data/best_model.ckpt")
    # args.add_argument('--dataset_path', type=str, default="/home/chan/data/augmented_dataset_300k.csv")
    # args.add_argument('--image_dir', type=str, default="/home/chan/data/augmented_images_300k/")
    args.add_argument('--trainset_path', type=str, default="gs://chandradeep_data/trainset_300k.record")
    args.add_argument('--valset_path', type=str, default="gs://chandradeep_data/valset_300k.record")

    config = args.parse_args()

    tf.keras.backend.clear_session()
    TPU_WORKER = 'grpc://' + '10.120.205.234:8470'
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver(TPU_WORKER)
    tf.config.experimental_connect_to_cluster(resolver)
    tf.tpu.experimental.initialize_tpu_system(resolver)
    strategy = tf.distribute.experimental.TPUStrategy(resolver)
#    tf.compat.v1.disable_eager_execution() 

    with strategy.scope():
        # model = BlazeFace(config).build_model()
        model = MobileNetV2(config).build_model()
        print(model)
        opt = tf.compat.v1.train.AdamOptimizer(learning_rate = config.learning_rate)
        model.compile(loss= tf.keras.losses.MeanSquaredError(), optimizer=opt)
    early_stopping = tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss', min_delta=0, patience=10, verbose=1, mode='min',
                    baseline=None, restore_best_weights=True
                )
    lr_scheduler = tf.keras.callbacks.LearningRateScheduler(scheduler)
    cbk = CustomModelCheckpoint()

    train_gen = create_dataset(config.train_batch_size, config.trainset_path, config.shuffle_buffer)
    STEP_SIZE_TRAIN = int((0.95 * config.num_data) // config.train_batch_size)
    val_gen = create_dataset(config.val_batch_size, config.valset_path, None)
    STEP_SIZE_VAL = int((0.05 * config.num_data) // config.val_batch_size)

    model.fit(x=train_gen, epochs = config.epochs, 
                                    steps_per_epoch = STEP_SIZE_TRAIN,
                                    validation_data = train_gen,
                                    validation_steps = STEP_SIZE_VAL,
                                    callbacks = [early_stopping, cbk])
