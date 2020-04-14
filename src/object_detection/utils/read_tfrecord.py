import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

def _parse_function(proto):
    # define your tfrecord again. Remember that you saved your image as a string.
    keys_to_features = {'image': tf.io.FixedLenFeature([], tf.string),
                        'l_prob':  tf.io.FixedLenFeature([], tf.float32),
                        'r_prob':  tf.io.FixedLenFeature([], tf.float32),
                        'l_center_x':  tf.io.FixedLenFeature([], tf.float32),
                        'l_center_y':  tf.io.FixedLenFeature([], tf.float32),
                        'l_width':  tf.io.FixedLenFeature([], tf.float32),
                        'l_height':  tf.io.FixedLenFeature([], tf.float32),
                        'r_center_x':  tf.io.FixedLenFeature([], tf.float32),
                        'r_center_y':  tf.io.FixedLenFeature([], tf.float32),
                        'r_width':  tf.io.FixedLenFeature([], tf.float32),
                        'r_height':  tf.io.FixedLenFeature([], tf.float32)}
    
    # Load one example
    parsed_features = tf.io.parse_single_example(proto, keys_to_features)
    all_features = [parsed_features['l_prob'],parsed_features['r_prob'],parsed_features['l_center_x'],parsed_features['l_center_y']
                    ,parsed_features['l_width'],parsed_features['l_height'],parsed_features['r_center_x'],parsed_features['r_center_y'],
                    parsed_features['r_width'],parsed_features['r_height']]
    # Turn your saved image string into an array
    parsed_features['image'] = tf.io.decode_raw(
        parsed_features['image'], tf.uint8)
    return parsed_features['image'], all_features

def create_dataset(batch_size, filepath, shuffle_buffer):
    dataset = tf.data.TFRecordDataset(filepath)
    dataset = dataset.map(_parse_function)
    dataset = dataset.repeat()
    dataset = dataset.shuffle(shuffle_buffer)
    dataset = dataset.batch(batch_size)
    iterator = iter(dataset)
    image, labels= iterator.get_next()
    image = tf.reshape(image, [-1, 224, 224, 3])
    return image, labels

if __name__ == "__main__":
    dataset = create_dataset("/home/noldsoul/Desktop/Uplara/keras_object_detection/src/object_detection/utils/trainset.record")
    iterator = iter(dataset)
    for i in range(50):
        image, labels= iterator.get_next()
        image = tf.reshape(image, [-1, 224, 224, 3])
        print(image.shape)
        image = np.squeeze(image, axis = 0)
        print(labels.shape)
        plt.imshow(image)
        plt.show()