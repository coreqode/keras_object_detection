import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import cv2

def _parse_function(proto):
    # define your tfrecord again. Remember that you saved your image as a string.
    keys_to_features = {'image': tf.io.FixedLenFeature([], tf.string),
                        'points':  tf.io.FixedLenFeature([50], tf.float32)}
    # Load one example
    parsed_features = tf.io.parse_single_example(proto, keys_to_features)
    all_features = parsed_features['points']
    # Turn your saved image string into an array
    image = tf.io.decode_jpeg(
        parsed_features['image'])
    image = tf.cast(tf.reshape(image, (224, 224, 3)), tf.float32)
    image = (image / 127.5) - 1 
    return image, all_features

def create_dataset(batch_size, filepath, shuffle_buffer):
    dataset = tf.data.TFRecordDataset(filepath)
    dataset = dataset.map(_parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.repeat()
    dataset = dataset.shuffle(shuffle_buffer)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    # iterator = iter(dataset)
    # image, labels= iterator.get_next()
    return dataset

if __name__ == "__main__":
    dataset = create_dataset(4, "/home/noldsoul/Desktop/Uplara/keras_object_detection/src/keypoint_regression/utils/trainset_300k.record", 10)
    for image, label in dataset.take(50):
        image = np.array(image[0])
        image = cv2.normalize(image, None, 255,0, cv2.NORM_MINMAX, cv2.CV_8UC1)
        cv2.imshow('image', image   )
        cv2.waitKey()
