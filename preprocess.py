import tensorflow as tf
import numpy as np
import glob
# use following commands when 'Segmentation fault' error occurs
# import matplotlib
# matplotlib.use('TkAgg') 
from matplotlib import pyplot as plt
from PIL import Image
import os
import random

def _bytes_feature(value):
    """ Returns a bytes_list from a string/byte"""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """ Returns a float_list from a float/double """
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """ Returns a int64_list from a bool/enum/int/uint """
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _image_as_bytes(imagefile):
    image = np.array(Image.open(imagefile))
    image_raw = image.tostring()
    return image_raw

def make_example(img, lab):
    """ TODO: Return serialized Example from img, lab """
    feature = {'encoded': _bytes_feature(img),
               'label': _int64_feature(lab)}

    example = tf.train.Example(features=tf.train.Features(feature=feature))
    return example.SerializeTostring()


def write_tfrecord(imagedir, datadir):
    """ TODO: write a tfrecord file containing img-lab pairs
        imagedir: directory of input images
        datadir: directory of output a tfrecord file (or multiple tfrecord files) """
    samples = []

    for label_name in os.listdir(imagedir):
        img_paths = glob.glob(os.path.join(imagedir, label_name, '*.png'))
        for img_path in img_paths:
            samples.append((img_path, label_name))
    

    if os.path.basename(imagedir) == 'test':
        mode = 'test'
    else:
        mode = 'train'

    if mode == 'test':
        samples = sorted(samples)

        os.makedirs(os.path.join(datadir, os.pardir, 'test_tfrecord'), exist_ok=True)
        test_writer = tf.python_io.TFRecordWriter(os.path.join(datadir, os.pardir, 'test_tfrecord', 'mnist.tfrecord'))

        for img_path, label_name in samples:
            # img_data = open(img_path, 'rb').read()
            img_data = _image_as_bytes(img_path)
            lab = int(label_name)
            example = make_example(img_data, lab)
            test_writer.write(example)
        test_writer.close()

        print("finish test_write_tfrecord")



    else:
        random.shuffle(samples)
        os.makedirs(os.path.join(datadir), exist_ok=True)
        os.makedirs(os.path.join(datadir, os.pardir, 'val_tfrecord'), exist_ok=True)

        # writer = tf.python_io.TFRecordWriter(datadir)
        tr_writer = tf.python_io.TFRecordWriter(os.path.join(datadir, 'mnist.tfrecord'))
        val_writer = tf.python_io.TFRecordWriter(os.path.join(datadir, os.pardir, 'val_tfrecord', 'mnist.tfrecord'))
        
        val_num = 0

        for img_path, label_name in samples:
            # img_data = open(img_path, 'rb').read()
            img_data = _image_as_bytes(img_path)
            lab = int(label_name)
            example = make_example(img_data, lab)
            
            val_num += 1

            if val_num < len(samples)//10:
                val_writer.write(example)
            else:
                tr_writer.write(example)

        tr_writer.close()
        val_writer.close()

        print("finish train_write_tfrecord")


def read_tfrecord(folder, batch=100, epoch=1):
    """ TODO: read tfrecord files in folder, Return shuffled mini-batch img,lab pairs
    img: float, 0.0~1.0 normalized
    lab: dim 10 one-hot vectors
    folder: directory where tfrecord files are stored in
    epoch: maximum epochs to train, default: 1 """
    filenames = glob.glob(os.path.join(folder, '*.tfrecord'))
    filename_queue = tf.train.string_input_producer(filenames, num_epochs=epoch)

    reader = tf.TFRecordReader()
    key, serialized_example = reader.read(filename_queue)

    key_to_feature = {'encoded' : tf.FixedLenFeature([], tf.string, default_value=''),
                      'label' : tf.FixedLenFeature([], tf.int64, default_value=0.)}

    features = tf.parse_single_example(serialized_example, features= key_to_feature)

    img = tf.decoded_raw(features['encoded'], tf.uint8)
    img = tf.reshape(img, [28, 28, 1])
    img /= 255
    lab = tf.cast(features['label'], tf.int32)
    lab = tf.one_hot(lab, 10)

    img, lab = tf.train.shuffle_batch([img, lab], batch_size=batch)

    return img, lab
