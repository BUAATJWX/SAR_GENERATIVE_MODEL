# Some code was borrowed from https://github.com/petewarden/tensorflow_makefile/blob/master/tensorflow/models/image/mnist/convolutional.py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import copy
import numpy
from scipy import ndimage
from scipy.misc import imsave
from six.moves import urllib

import tensorflow as tf
import scipy.io as sio
SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'
DATA_DIRECTORY = "data"

# Params for MNIST
IMAGE_SIZE = 64
NUM_CHANNELS = 1
PIXEL_DEPTH = 1.0#10.0
NUM_LABELS = 10 #22+1
VALIDATION_SIZE = 5000  # Size of the validation set.



# Download MNIST data
def maybe_download(filename):
    """Download the data from Yann's website, unless it's already here."""
    if not tf.gfile.Exists(DATA_DIRECTORY):
        tf.gfile.MakeDirs(DATA_DIRECTORY)
    filepath = os.path.join(DATA_DIRECTORY, filename)
    if not tf.gfile.Exists(filepath):
        filepath, _ = urllib.request.urlretrieve(SOURCE_URL + filename, filepath)
        with tf.gfile.GFile(filepath) as f:
            size = f.size()
        print('Successfully downloaded', filename, size, 'bytes.')
    return filepath

def extractdb_images(filename,tag=4):
  #print('Extracting', filename)
  imdb = sio.loadmat(filename)
  if tag==1:
    data = imdb.get('trainData')
  elif tag==2:
    data = imdb.get('valData')
  elif tag==3:
    data = imdb.get('testData')
  elif tag==4:
    data = imdb.get('trainimdb')
  else:
    data = imdb.get('newimdb')
  # if tag<5:
  #   mean_data=numpy.mean(data)
  #   data=data-numpy.mean(data)
  data = numpy.asarray(data)
  data = data.reshape(data.shape[0], IMAGE_SIZE, IMAGE_SIZE, 1)
  return data

# Extract the images
def extract_data(filename, norm_shift=False, norm_scale=True,tag=1):
    """Extract the images into a 4D tensor [image index, y, x, channels].
    Values are rescaled from [0, 255] down to [-0.5, 0.5].
    """
    print('Extracting', filename)
    data = extractdb_images(filename, tag)

    if norm_shift:
      data = data - (PIXEL_DEPTH / 2.0)
    if norm_scale:
      data = data / PIXEL_DEPTH
    #data = data.reshape(num_images, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS)
    num=data.shape[0]
    data = numpy.reshape(data, [num, -1])
    return data

def dense_to_one_hot(labels_dense, num_classes=10):
  """Convert class labels from scalars to one-hot vectors."""
  num_labels = labels_dense.shape[0]
  index_offset = numpy.arange(num_labels) * num_classes
  labels_one_hot = numpy.zeros((num_labels, num_classes))
  labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
  return labels_one_hot

def extractdb_labels(filename, tag,one_hot=True):
  #print('Extracting', filename)
  imdb = sio.loadmat(filename)
  if tag == 1:
    labels = imdb.get('trainLabel')
  elif tag == 2:
    labels= imdb.get('valLabel')
  else:
    labels= imdb.get('testLabel')
  labels= numpy.asarray(labels)
  labels=labels-1
  if one_hot:
    return dense_to_one_hot(labels)
  return labels

# Extract the labels
def extract_labels(filename, tag):
    """Extract the labels into a vector of int64 label IDs."""
    print('Extracting', filename)
    return extractdb_labels(filename, tag, one_hot=True)


# Augment training data
def expend_training_data(images, labels):

    expanded_images = []
    expanded_labels = []

    j = 0 # counter
    for x, y in zip(images, labels):
        j = j+1
        if j%100==0:
            print ('expanding data : %03d / %03d' % (j,numpy.size(images,0)))

        # register original data
        expanded_images.append(x)
        expanded_labels.append(y)

        # get a value for the background
        # zero is the expected value, but median() is used to estimate background's value
        bg_value = numpy.median(x) # this is regarded as background's value
        #image = numpy.reshape(x, (IMAGE_SIZE, IMAGE_SIZE))
        image = numpy.reshape(x, (96, 96))


        for i in range(4):
            # rotate the image with random degree
            angle = numpy.random.randint(-15,15,1)
            new_img = ndimage.rotate(image,angle,reshape=False, cval=bg_value)

            # shift the image with random distance
            shift = numpy.random.randint(-4, 4, 2)
            new_img_ = ndimage.shift(new_img,shift, cval=bg_value)

            # register new training data
            expanded_images.append(numpy.reshape(new_img_, IMAGE_SIZE**2))
            expanded_labels.append(y)

    # images and labels are concatenated for random-shuffle at each epoch
    # notice that pair of image and label should not be broken
    expanded_train_total_data = numpy.concatenate((expanded_images, expanded_labels), axis=1)
    numpy.random.shuffle(expanded_train_total_data)

    return expanded_train_total_data

def expend_total_data(images, labels,tag=True):
    expanded_images = []
    expanded_labels = []

    j = 0  # counter
    for x, y in zip(images, labels):
        j = j + 1
        if j % 100 == 0:
            print('expanding data : %03d / %03d' % (j, numpy.size(images, 0)))

        image = numpy.reshape(x, (96, 96))
        cent_image = image[16:80, 16:80]

        # register original data
        expanded_images.append(numpy.reshape(cent_image,64**2))
        expanded_labels.append(y)

        # get a value for the background
        # zero is the expected value, but median() is used to estimate background's value
        bg_value = numpy.median(x)  # this is regarded as background's value
        # image = numpy.reshape(x, (IMAGE_SIZE, IMAGE_SIZE))

        if tag:
         for i in range(2):
            # rotate the image with random degree
            angle = numpy.random.randint(-15, 15, 1)
            new_img_ = ndimage.rotate(image, angle, reshape=False, cval=bg_value)
            new_img_ = new_img_[16:80, 16:80]
            # shift the image with random distance
            #down_shift = numpy.random.randint(12, 22, 1)[0]
            #right_shift = numpy.random.randint(10, 24, 1)
            #new_img_ = image[down_shift:down_shift+64,down_shift:down_shift+64]

            # register new training data
            expanded_images.append(numpy.reshape(new_img_, 64 ** 2))
            expanded_labels.append(y)

    # images and labels are concatenated for random-shuffle at each epoch
    # notice that pair of image and label should not be broken
    expanded_train_total_data = numpy.concatenate((expanded_images, expanded_labels), axis=1)
    numpy.random.shuffle(expanded_train_total_data)

    return expanded_train_total_data

# Prepare MNISt data
def prepare_MNIST_data(use_norm_shift=False, use_norm_scale=True, use_data_augmentation=False):
    # Get the data.


    total_data = extract_data("./data/trainData", use_norm_shift, use_norm_scale, 1)
    total_data_len=total_data.shape[0]
    total_labels = extract_labels("./data/trainLabel", tag=1)

    AangleDb = sio.loadmat("./data/trainAangle")
    trainAangle = AangleDb.get('trainAangle')
    sin_Aangle = numpy.sin(trainAangle/180.0*numpy.pi)
    cos_Aangle = numpy.cos(trainAangle/180.0*numpy.pi)

    total_data= numpy.reshape(total_data, [total_data_len, -1])


    index=numpy.arange(0,128)
    tmp=total_data[index,:]
    test_data=copy.deepcopy(tmp)
    tmp = total_labels[index, :]
    test_labels = copy.deepcopy(tmp)
    tmpsin =sin_Aangle[index, :]
    tmpsin = copy.deepcopy(tmpsin)
    tmpcos = cos_Aangle[index, :]
    tmpcos = copy.deepcopy(tmpcos)

    test_labels = numpy.concatenate((test_labels, tmpsin), axis=1)
    test_labels = numpy.concatenate((test_labels, tmpcos), axis=1)



    if use_data_augmentation:
        train_total_data = expend_total_data(total_data, total_labels,False)
    else:
        train_total_data = numpy.concatenate((total_data, total_labels), axis=1)

    train_total_data = numpy.concatenate((train_total_data, sin_Aangle), axis=1)
    train_total_data = numpy.concatenate((train_total_data, cos_Aangle), axis=1)


    train_size = train_total_data.shape[0]
    print('training data size: %03d ' % (train_size))
    return train_total_data, train_size, test_data, test_labels






