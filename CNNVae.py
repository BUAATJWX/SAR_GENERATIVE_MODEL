
import os, sys
sys.path.append(os.getcwd())

import time
import functools

import numpy as np
import tensorflow as tf
#import sklearn.datasets

import tflib as lib
import tflib.ops.linear
import tflib.ops.conv2d
import tflib.ops.batchnorm
import tflib.ops.deconv2d
import tflib.save_images
import tflib.small_imagenet
import tflib.ops.layernorm
import tflib.plot
#import sar_data

DATA_DIR = './IMAGENET_DATA'
if len(DATA_DIR) == 0:
    raise Exception('Please specify path to data directory in gan_64x64.py!')

MODE = 'wgan-gp' # dcgan, wgan, wgan-gp, lsgan
DIM = 32 # Model dimensionality
NOISE_DIM=64
LABEL_DIM=10+2
Z_DIM=64
CRITIC_ITERS = 5 # How many iterations to train the critic for
N_GPUS = 1 # Number of GPUs
BATCH_SIZE = 128 # Batch size. Must be a multiple of N_GPUS
ITERS = 100000 # How many iterations to train for
LAMBDA = 10 # Gradient penalty lambda hyperparameter
OUTPUT_DIM = 64*64*1 # Number of pixels in each iamge

lib.print_model_settings(locals().copy())

def conv_cond_concat(x, y):
    """Concatenate conditioning vector on feature map axis."""
    x_shapes = x.get_shape()
    y_shapes = y.get_shape()
    #y=tf.reshape(y,[x_shapes[0], y_shapes[1],1,1])
    y=y * tf.ones([x_shapes[0], y_shapes[1], x_shapes[2], x_shapes[3]])
    return tf.concat([x , y], 1)

def LeakyReLU(x, alpha=0.2):
    return tf.maximum(alpha*x, x)

def ReLULayer(name, n_in, n_out, inputs):
    output = lib.ops.linear.Linear(name+'.Linear', n_in, n_out, inputs, initialization='he')
    return tf.nn.relu(output)

def LeakyReLULayer(name, n_in, n_out, inputs):
    output = lib.ops.linear.Linear(name+'.Linear', n_in, n_out, inputs, initialization='he')
    return LeakyReLU(output)

def Normalize(name, axes, inputs):
    if ('Discriminator' in name) and (MODE == 'wgan-gp'):
        if axes != [0,2,3]:
            raise Exception('Layernorm over non-standard axes is unsupported')
        return lib.ops.layernorm.Layernorm(name,[1,2,3],inputs)
    else:
        return lib.ops.batchnorm.Batchnorm(name,axes,inputs,fused=True)

# def Normalize(name, axes, inputs):
#     # if ('Discriminator' in name) and (MODE == 'wgan-gp'):
#     #     if axes != [0,2,3]:
#     #         raise Exception('Layernorm over non-standard axes is unsupported')
#     return lib.ops.layernorm.Layernorm(name,[1,2,3],inputs)
#     # else:
#         #return lib.ops.batchnorm.Batchnorm(name,axes,inputs,fused=True)

def pixcnn_gated_nonlinearity(a, b):
    return tf.sigmoid(a) * tf.tanh(b)

def SubpixelConv2D(*args, **kwargs):
    kwargs['output_dim'] = 4*kwargs['output_dim']
    output = lib.ops.conv2d.Conv2D(*args, **kwargs)
    output = tf.transpose(output, [0,2,3,1])
    output = tf.depth_to_space(output, 2)
    output = tf.transpose(output, [0,3,1,2])
    return output

def ConvMeanPool(name, input_dim, output_dim, filter_size, inputs, he_init=True, biases=True):
    output = lib.ops.conv2d.Conv2D(name, input_dim, output_dim, filter_size, inputs, he_init=he_init, biases=biases)
    output = tf.add_n([output[:,:,::2,::2], output[:,:,1::2,::2], output[:,:,::2,1::2], output[:,:,1::2,1::2]]) / 4.
    return output

def MeanPoolConv(name, input_dim, output_dim, filter_size, inputs, he_init=True, biases=True):
    output = inputs
    output = tf.add_n([output[:,:,::2,::2], output[:,:,1::2,::2], output[:,:,::2,1::2], output[:,:,1::2,1::2]]) / 4.
    output = lib.ops.conv2d.Conv2D(name, input_dim, output_dim, filter_size, output, he_init=he_init, biases=biases)
    return output

def UpsampleConv(name, input_dim, output_dim, filter_size, inputs, he_init=True, biases=True):
    output = inputs
    output = tf.concat([output, output, output, output], axis=1)
    output = tf.transpose(output, [0,2,3,1])
    output = tf.depth_to_space(output, 2)
    output = tf.transpose(output, [0,3,1,2])
    output = lib.ops.conv2d.Conv2D(name, input_dim, output_dim, filter_size, output, he_init=he_init, biases=biases)
    return output

def ResidualBlock(name, input_dim, output_dim, filter_size, inputs, resample=None, he_init=True):
    """
    resample: None, 'down', or 'up'
    """
    if resample=='down':
        conv_shortcut = MeanPoolConv
        conv_1        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=input_dim, output_dim=input_dim)
        conv_2        = functools.partial(ConvMeanPool, input_dim=input_dim, output_dim=output_dim)
    elif resample=='up':
        conv_shortcut = UpsampleConv
        conv_1        = functools.partial(UpsampleConv, input_dim=input_dim, output_dim=output_dim)
        conv_2        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=output_dim, output_dim=output_dim)
    elif resample==None or resample=='down_none':
        conv_shortcut = lib.ops.conv2d.Conv2D
        conv_1        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=input_dim,  output_dim=input_dim)
        conv_2        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=input_dim, output_dim=output_dim)
    else:
        raise Exception('invalid resample value')

    if output_dim==input_dim and (resample==None or resample=='down_none'):
        shortcut = inputs # Identity skip-connection
    else:
        shortcut = conv_shortcut(name+'.Shortcut', input_dim=input_dim, output_dim=output_dim, filter_size=1,
                                 he_init=False, biases=True, inputs=inputs)
    output = inputs
    if resample=='down' or resample=='down_none':
      output = Normalize(name+'.BN1', [0,2,3], output)
    output = LeakyReLU(output)
    #output = tf.nn.relu(output)
    output = conv_1(name+'.Conv1', filter_size=filter_size, inputs=output, he_init=he_init, biases=False)
    if resample == 'down' or resample=='down_none':
      output = Normalize(name+'.BN2', [0,2,3], output)
    output = LeakyReLU(output)
    #output = tf.nn.relu(output)
    output = conv_2(name+'.Conv2', filter_size=filter_size, inputs=output, he_init=he_init)

    return shortcut + output

def decode_inference(noise=None, labels=None,dim=DIM, nonlinearity=tf.nn.relu):
    # if noise is None:
    #     noise = tf.random_normal([n_samples, 128])
    noise = tf.concat([noise, labels], 1)
    output = lib.ops.linear.Linear('Generator.Input', NOISE_DIM+LABEL_DIM, 4*4*8*dim, noise)
    output = tf.reshape(output, [-1, 8*dim, 4, 4])

    #output = ResidualBlock('Generator.Res1', 8*dim, 8*dim, 3, output, resample='up')
    #output = ResidualBlock('Generator.Res11', 8 * dim, 8 * dim, 3, output)
    output = ResidualBlock('Generator.Res2', 8*dim, 4*dim, 3, output, resample='up')
    output = ResidualBlock('Generator.Res21', 4 * dim, 4 * dim, 3, output)
    output = ResidualBlock('Generator.Res3', 4*dim, 2*dim, 3, output, resample='up')
    output = ResidualBlock('Generator.Res31', 2 * dim, 2 * dim, 3, output)
    output = ResidualBlock('Generator.Res4', 2*dim, 1*dim, 3, output, resample='up')
    output = ResidualBlock('Generator.Res41', 1 * dim, 1 * dim, 3, output)
    #output = Normalize('Generator.OutputN', [0,2,3], output)
    #output = tf.nn.relu(output)
    output = LeakyReLU(output)  # wangke
    #output = lib.ops.conv2d.Conv2D('Generator.Output', 1*dim, 1, 3, output)
    kernel = tf.get_variable('Generator.Res5.kernel', [5, 5, 1, dim], initializer= tf.truncated_normal_initializer(stddev=0.02))
    #biases = tf.get_variable('Generator.Res5.', [1], initializer=b_init)
    output = tf.transpose(output, [0, 2, 3, 1])
    output = tf.nn.conv2d_transpose(output, kernel, output_shape=[BATCH_SIZE, 64, 64, 1], strides=[1, 2, 2, 1],padding='SAME')
    output = tf.tanh(output)

    return tf.reshape(output, [-1, OUTPUT_DIM])

def encode_inference(inputs, labels, dim,dim_z):

    output = tf.reshape(inputs, [BATCH_SIZE, 1, 64, 64])
    #output = tf.reshape(inputs, [-1,64, 64,1])
    labels = tf.reshape(labels, shape=[BATCH_SIZE, LABEL_DIM,1,1])
    output = conv_cond_concat(output, labels)
    output = lib.ops.conv2d.Conv2D('Discriminator.Input', 1+LABEL_DIM, dim, 3, output, he_init=False)

    output = ResidualBlock('Discriminator.Res1', dim, 2*dim, 3, output, resample='down')
    output = ResidualBlock('Discriminator.Res11', 2*dim, 2*dim, 3, output, resample='down_none')
    output = ResidualBlock('Discriminator.Res2', 2*dim, 4*dim, 3, output, resample='down')
    output = ResidualBlock('Discriminator.Res21', 4 * dim, 4 * dim, 3, output,resample='down_none')
    output = ResidualBlock('Discriminator.Res3', 4*dim, 8*dim, 3, output, resample='down')
    output = ResidualBlock('Discriminator.Res31',8*dim, 8*dim, 3, output,resample='down_none')
    output = ResidualBlock('Discriminator.Res4', 8*dim, 8*dim, 3, output, resample='down')
    output = ResidualBlock('Discriminator.Res41',8*dim, 8*dim, 3, output,resample='down_none')
    # output = tf.reshape(output, [-1, 4*4*8*dim])
    # output = lib.ops.linear.Linear('Discriminator.Output', 4 * 4 * 8 * dim, dim_z , output)
    # return LeakyReLU(output)
    #return tf.reshape(output, [-1])
    output = LeakyReLU(output)
    output = tf.reduce_mean(output, [2, 3])
    output = tf.reshape(output, [-1, 8*dim])
    output = lib.ops.linear.Linear('Discriminator.Output', 8 * dim, dim_z , output)
    return output



def gaussian_CNN_encoder(x, cond_info, dim_z):
    with tf.variable_scope("g_encoder_"):
        gaussian_params = encode_inference(x,cond_info,DIM,dim_z)

        return gaussian_params

def gaussian_CNN_decoder(z, cond_info,reuse=False):

    with tf.variable_scope("g_decoder_",reuse=reuse):
      y=10.0*decode_inference(noise=z, labels=cond_info, dim=DIM, nonlinearity=tf.nn.relu)
      return y


def CNN_decoder(z, cond_info):

    y = gaussian_CNN_decoder(z, cond_info,reuse=True)

    return y


def discriminator(x,n_hidden=256):
    with tf.variable_scope("g_dis_"):
        # initializers
        w_init = tf.contrib.layers.variance_scaling_initializer()
        #w_init=tf.truncated_normal_initializer(stddev=0.001)
        b_init = tf.constant_initializer(0.)

        # 1st hidden layer
        w0 = tf.get_variable('weight0', [x.get_shape()[1], n_hidden], initializer=w_init)
        b0 = tf.get_variable('b0', [n_hidden], initializer=b_init)
        h0 = tf.matmul(x, w0)+b0
        #h0 = Normalize('discriminator.Output1', [0, 2, 3], h0)
        #h0 = tf.layers.batch_normalization(h0, training=True)
        h0 = tf.nn.relu(h0)

        # 2nd hidden layer
        w1 = tf.get_variable('weight1', [h0.get_shape()[1], n_hidden], initializer=w_init)
        b1 = tf.get_variable('b1', [n_hidden], initializer=b_init)
        h1 = tf.matmul(h0, w1)+b1
        #h1 = Normalize('discriminator.Output2', [0, 2, 3], h1)
        #h1 = tf.layers.batch_normalization(h1, training=True)
        h1 = tf.nn.relu(h1)

        #3nd hidden layer
        w2 = tf.get_variable('weight2', [h1.get_shape()[1], n_hidden], initializer=w_init)
        b2 = tf.get_variable('b2', [n_hidden], initializer=b_init)
        h2 = tf.matmul(h1, w2)+b2
        #h2 = Normalize('discriminator.Output3', [0, 2, 3], h2)
        #h2 = tf.layers.batch_normalization(h2, training=True)
        h2 = tf.nn.relu(h2)

        # 4nd hidden layer
        w3 = tf.get_variable('weight3', [h2.get_shape()[1], 1], initializer=w_init)
        b3 = tf.get_variable('b3', [1], initializer=b_init)
        h3 = tf.matmul(h2, w3) + b3
        return h3

        # prob = tf.nn.sigmoid(h2)
        # return prob
        #return  h2