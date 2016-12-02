import tensorflow as tf
from tensorflow.python.framework import ops, dtypes
import nibabel
import numpy as np
import os, csv, glob, random

data_dir = os.getcwd()
labels_file = 'oasis_cross_sectional.csv'
n = 80
test_set_size = 20
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
IMAGE_DEPTH = 160
NUM_CHANNELS = 1
BATCH_SIZE = 10


def encode_label(label):
    if len(label) == 0:
        return 0.0
    else:
        return float(label)


def read_label_file(file):
    filepaths = []
    labels = []
    with open(file) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            filepaths.append(row['ID'])
            labels.append(encode_label(row['CDR']))
    return filepaths, labels


def get_data(filenames):
    data_dir = os.getcwd()
    data = []
    for filename in filenames:
        path = data_dir + '/data/' + filename + '/PROCESSED/MPRAGE/SUBJ_111/'
        if os.path.isdir(path):
            for file in os.listdir(path):
                if file.endswith('.hdr'):
                    full_path = path + file
                    image = np.array(nibabel.analyze.load(full_path).get_data())
                    image = np.reshape(image, (IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_DEPTH))
                    data.append(image)
    return data


# Load images and label data
filenames, labels = read_label_file(labels_file)
all_filenames = filenames[:n]
all_images = get_data(all_filenames)
all_labels = labels[:n]

# Start building input pipeline
all_images = ops.convert_to_tensor(all_filenames, dtype=tf.string)
all_labels = ops.convert_to_tensor(all_labels, dtype=tf.float32)

# Partition the data
partitions = [0] * len(all_filenames)
partitions[:test_set_size] = [1] * test_set_size
random.shuffle(partitions)
train_images, test_images = tf.dynamic_partition(all_images, partitions, 2)
train_labels, test_labels = tf.dynamic_partition(all_labels, partitions, 2)

# Build input queues and define how to load images
train_input_queue = tf.train.slice_input_producer(
    [train_images, train_labels], shuffle=False)
test_input_queue = tf.train.slice_input_producer(
    [test_images, test_labels], shuffle=False)

train_image = train_input_queue[0]
train_label = train_input_queue[1]

test_image = test_input_queue[0]
test_label = test_input_queue[1]

# Group samples into batches
train_image_batch, train_label_batch = tf.train.batch(
    [train_image, train_label],
    batch_size=BATCH_SIZE)
test_image_batch, test_label_batch = tf.train.batch(
    [test_image, test_label],
    batch_size=BATCH_SIZE)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    # print("".join([chr(item) for item in file_content.eval()]))
    print('from the train set:')
    for i in range(n - test_set_size):
        print(sess.run(train_label_batch))

    print('from the test set:')
    for i in range(10):
        print(sess.run(test_label_batch))

    coord.request_stop()
    coord.join(threads)
    sess.close()
