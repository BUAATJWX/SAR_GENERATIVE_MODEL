import tensorflow as tf
import numpy as np
import mnist_data
import os
#import vae
import plot_utils
import glob
import CNNVae
import argparse
import scipy.io as sio
save_result = True
IMAGE_SIZE_MNIST = 64



# losses
def gradient_penalty(real, fake, f):
    def interpolate(a, b):
        shape = tf.concat((tf.shape(a)[0:1], tf.tile([1], [a.shape.ndims - 1])), axis=0)
        alpha = tf.random_uniform(shape=shape, minval=0., maxval=1.)
        inter = a + alpha * (b - a)
        inter.set_shape(a.get_shape().as_list())
        return inter

    x = interpolate(real, fake)
    pred = f(x)
    gradients = tf.gradients(pred, x)[0]
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
    gp = tf.reduce_mean((slopes - 1.) ** 2)
    return gp

def num_to_one_hot(labels_dense, num_classes=10):
  """Convert class labels from scalars to one-hot vectors."""
  num_labels = labels_dense.shape[0]
  index_offset = np.arange(num_labels) * num_classes
  labels_one_hot = np.zeros((num_labels, num_classes))
  labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
  return labels_one_hot

"""parsing and configuration"""
def parse_args():
    desc = "Tensorflow implementation of 'SAR Generative model '"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--results_path', type=str, default='results',
                        help='File path of output images')

    parser.add_argument('--add_noise', type=bool, default=False, help='Boolean for adding salt & pepper noise to input image')

    parser.add_argument('--dim_z', type=int, default='64', help='Dimension of latent vector')

    parser.add_argument('--n_hidden', type=int, default=256, help='Number of hidden units in MLP')

    parser.add_argument('--learn_rate', type=float, default=0.0002, help='Learning rate for Adam optimizer')

    parser.add_argument('--num_epochs', type=int, default=1000, help='The number of epochs to run')

    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')

    parser.add_argument('--PRR', type=bool, default=True,
                        help='Boolean for plot-reproduce-result')

    parser.add_argument('--PRR_n_img_x', type=int, default=16,
                        help='Number of images along x-axis')

    parser.add_argument('--PRR_n_img_y', type=int, default=8,
                        help='Number of images along y-axis')

    parser.add_argument('--PRR_resize_factor', type=float, default=1.0,
                        help='Resize factor for each displayed image')

    parser.add_argument('--PMLR', type=bool, default=False,
                        help='Boolean for plot-manifold-learning-result')

    parser.add_argument('--PMLR_n_img_x', type=int, default=20,
                        help='Number of images along x-axis')

    parser.add_argument('--PMLR_n_img_y', type=int, default=20,
                        help='Number of images along y-axis')

    parser.add_argument('--PMLR_resize_factor', type=float, default=1.0,
                        help='Resize factor for each displayed image')

    parser.add_argument('--PMLR_z_range', type=float, default=2.0,
                        help='Range for unifomly distributed latent vector')

    parser.add_argument('--PMLR_n_samples', type=int, default=5000,
                        help='Number of samples in order to get distribution of labeled data')

    return check_args(parser.parse_args())

"""checking arguments"""
def check_args(args):

    # --results_path
    # try:
    #     os.mkdir(args.results_path)
    # except (FileExistsError):
    #     pass

    if not os.path.exists(args.results_path):
        os.mkdir(args.results_path)

    # delete all existing files
    files = glob.glob(args.results_path+'/*')
    for f in files:
        os.remove(f)

    # --add_noise
    try:
        assert args.add_noise == True or args.add_noise == False
    except:
        print('add_noise must be boolean type')
        return None

    # --dim-z
    try:
        assert args.dim_z > 0
    except:
        print('dim_z must be positive integer')
        return None

    # --n_hidden
    try:
        assert args.n_hidden >= 1
    except:
        print('number of hidden units must be larger than one')

    # --learn_rate
    try:
        assert args.learn_rate > 0
    except:
        print('learning rate must be positive')

    # --num_epochs
    try:
        assert args.num_epochs >= 1
    except:
        print('number of epochs must be larger than or equal to one')

    # --batch_size
    try:
        assert args.batch_size >= 1
    except:
        print('batch size must be larger than or equal to one')

    # --PRR
    try:
        assert args.PRR == True or args.PRR == False
    except:
        print('PRR must be boolean type')
        return None

    if args.PRR == True:
        # --PRR_n_img_x, --PRR_n_img_y
        try:
            assert args.PRR_n_img_x >= 1 and args.PRR_n_img_y >= 1
        except:
            print('PRR : number of images along each axis must be larger than or equal to one')

        # --PRR_resize_factor
        try:
            assert args.PRR_resize_factor > 0
        except:
            print('PRR : resize factor for each displayed image must be positive')

    # --PMLR
    try:
        assert args.PMLR == True or args.PMLR == False
    except:
        print('PMLR must be boolean type')
        return None

    if args.PMLR == True:
        try:
            assert args.dim_z == 2
        except:
            print('PMLR : dim_z must be two')

        # --PMLR_n_img_x, --PMLR_n_img_y
        try:
            assert args.PMLR_n_img_x >= 1 and args.PMLR_n_img_y >= 1
        except:
            print('PMLR : number of images along each axis must be larger than or equal to one')

        # --PMLR_resize_factor
        try:
            assert args.PMLR_resize_factor > 0
        except:
            print('PMLR : resize factor for each displayed image must be positive')

        # --PMLR_z_range
        try:
            assert args.PMLR_z_range > 0
        except:
            print('PMLR : range for unifomly distributed latent vector must be positive')

        # --PMLR_n_samples
        try:
            assert args.PMLR_n_samples > 100
        except:
            print('PMLR : Number of samples in order to get distribution of labeled data must be large enough')

    return args

"""main function"""
def main(args):

    """ parameters """
    RESULTS_DIR = args.results_path

    # network architecture
    ADD_NOISE = args.add_noise

    n_hidden = args.n_hidden
    dim_img = IMAGE_SIZE_MNIST**2  # number of pixels for a MNIST image
    dim_z = args.dim_z

    # train
    n_epochs = args.num_epochs
    batch_size = args.batch_size
    learn_rate = args.learn_rate

    # Plot
    PRR = args.PRR                              # Plot Reproduce Result
    PRR_n_img_x = args.PRR_n_img_x              # number of images along x-axis in a canvas
    PRR_n_img_y = args.PRR_n_img_y              # number of images along y-axis in a canvas
    PRR_resize_factor = args.PRR_resize_factor  # resize factor for each image in a canvas

    PMLR = args.PMLR                            # Plot Manifold Learning Result
    PMLR_n_img_x = args.PMLR_n_img_x            # number of images along x-axis in a canvas
    PMLR_n_img_y = args.PMLR_n_img_y            # number of images along y-axis in a canvas
    PMLR_resize_factor = args.PMLR_resize_factor# resize factor for each image in a canvas
    PMLR_z_range = args.PMLR_z_range            # range for random latent vector
    PMLR_n_samples = args.PMLR_n_samples        # number of labeled samples to plot a map from input data space to the latent space

    """ prepare MNIST data """

    train_total_data, train_size, test_data, test_labels = mnist_data.prepare_MNIST_data()
    n_samples = train_size
    para_lamda=10.0
    clipping_parameter = 0.01
    n_critic = 5
    #train_data = train_total_data[:, :-mnist_data.NUM_LABELS]

    """ build graph """

    # input placeholders
    # In denoising-autoencoder, x_hat == x + noise, otherwise x_hat == x
    #x_hat = tf.placeholder(tf.float32, shape=[None, dim_img], name='input_img')
    x = tf.placeholder(tf.float32, shape=[None, dim_img], name='target_img')
    batchsize=tf.placeholder(tf.float32,  name='batchsize')

    # input for PMLR
    z = tf.placeholder(tf.float32, shape=[None, dim_z], name='latent_variable')

    cond_info= tf.placeholder(tf.float32, shape=[None, 12], name='cond_info')


    encoder_output = CNNVae.gaussian_CNN_encoder(x, cond_info, dim_z)

    decoder_output = CNNVae.gaussian_CNN_decoder(encoder_output,cond_info)


    with tf.variable_scope('Discriminator') as scope:
        D_real = CNNVae.discriminator(z,n_hidden)
        scope.reuse_variables()
        D_fake = CNNVae.discriminator(encoder_output,n_hidden)

        alpha = tf.random_uniform(
            shape=[batch_size, 1],
            minval=0.,
            maxval=1.
        )
        differences = encoder_output - z  # may cause problem!!!
        interpolates = z + (alpha * differences)
        # gradients = tf.gradients(Discriminator(interpolates), [interpolates])[0]
        gradients = tf.gradients(CNNVae.discriminator(interpolates, n_hidden), [interpolates])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
        gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)
        ddx = 10.0 * gradient_penalty

        # ddx = gradient_penalty(z, encoder_output, CNNVae.discriminator)
        # ddx = ddx*10.0
    with tf.name_scope('Loss'):

        #marginal_likelihood = tf.reduce_sum(3.14/4.0*tf.exp(2.0*(x-decoder_output))-2.0*(x-decoder_output), 1)
        #loss_reconstr = tf.reduce_mean(marginal_likelihood)
        loss_reconstr = tf.reduce_mean(3.14/4.0*tf.exp(2.0*(x-decoder_output))-2.0*(x-decoder_output))
        # Adversarial loss to approx. Q(z|X)
        with tf.name_scope('Discriminator_loss'):
            #loss_discriminator = -para_lamda*(tf.reduce_mean(tf.log(D_real)) + tf.reduce_mean(tf.log(1.0-D_fake)))
            loss_discriminator = 1.0*(tf.reduce_mean(D_fake) - tf.reduce_mean(D_real)+ddx)

        with tf.name_scope('Encoder_loss'):
            #loss_encoder = -para_lamda* tf.reduce_mean(tf.log(D_fake))
            loss_encoder = -(1.0)*tf.reduce_mean(D_fake)

    vars = tf.trainable_variables()
    enc_params = [v for v in vars if 'g_encoder_' in v.name]
    dec_params = [v for v in vars if 'g_decoder_' in v.name]
    dis_params = [v for v in vars if 'g_dis_' in v.name]
    dis_weights = [w for w in dis_params if 'weight' in w.name]


    with tf.variable_scope('Discriminator_Accuracy'):
        accuracy_real = tf.reduce_mean(tf.cast(tf.greater_equal(D_real, 0.5), tf.float16))
        accuracy_fake = tf.reduce_mean(tf.cast(tf.less(D_fake, 0.5), tf.float16))
        accuracy_tot = (accuracy_real + accuracy_fake) / 2
        #accuracy_tot = tf.reduce_mean(D_real) - tf.reduce_mean(D_fake)
    #clipped_weights = clip_weights(dis_weights, clipping_parameter, 'clip_weights')
    CLIP = [-0.04, 0.04]
    clipped_weights = [var.assign(tf.clip_by_value(var, CLIP[0], CLIP[1])) for var in dis_weights]
    with tf.name_scope('Optimizer'):
        train_op_AE = tf.train.AdamOptimizer(learning_rate=learn_rate,beta1=0.,beta2=0.9).minimize(loss_reconstr+para_lamda*loss_encoder,var_list=[dec_params,enc_params])
        train_op_Dis = tf.train.AdamOptimizer(learning_rate=learn_rate,beta1=0.,beta2=0.9).minimize(para_lamda*loss_discriminator,var_list=[dis_params])


    test_smaple_size=12800
    test_batch_size=128
    z_test = tf.placeholder(tf.float32, shape=[test_batch_size, dim_z])
    test_cond_info = tf.placeholder(tf.float32, shape=[test_batch_size, 12], name='test_cond_info')
    test_op = CNNVae.CNN_decoder(z_test, test_cond_info)
    mu_test = tf.zeros([test_smaple_size, dim_z], dtype=tf.float32)
    test_sample = mu_test + tf.random_normal(tf.shape(mu_test), 0, 1, dtype=tf.float32)
    test_rand = np.random.randint(0, 10,size=[test_smaple_size,1]) #[0,10)
    test_info=num_to_one_hot(test_rand)
    test_angle=np.random.randint(0, 360,size=[test_smaple_size,1]) #[0,360)
    sin_angle=np.sin(test_angle/180.0*np.pi)
    cos_angle=np.cos(test_angle/180.0*np.pi)
    test_info = np.concatenate((test_info, sin_angle), axis=1)
    test_info = np.concatenate((test_info, cos_angle), axis=1)
    savename = "./label" + ".mat"
    sio.savemat(savename, {'label': test_rand})
    savename = "./angle" + ".mat"
    sio.savemat(savename, {'angle': test_angle})
    """ training """
    loss_array = np.zeros(shape=[n_epochs, 1], dtype=np.float32)
    epoch_array = np.zeros(shape=[n_epochs, 1], dtype=np.uint)

    # Plot for reproduce performance
    if PRR:
        PRR = plot_utils.Plot_Reproduce_Performance(RESULTS_DIR, PRR_n_img_x, PRR_n_img_y, IMAGE_SIZE_MNIST, IMAGE_SIZE_MNIST, PRR_resize_factor)

        x_PRR = test_data[0:PRR.n_tot_imgs, :]
        x_PRR_info = test_labels[0:PRR.n_tot_imgs, :]
        x_PRR_img = x_PRR.reshape(PRR.n_tot_imgs, IMAGE_SIZE_MNIST, IMAGE_SIZE_MNIST)
        PRR.save_images(x_PRR_img, name='input.jpg')
        sio.savemat('testimage.mat', {'testimage': x_PRR})


    # train
    total_batch = int(n_samples / batch_size)
    min_tot_loss = 1e99
    min_tot_mar_loss=1e99
    #force using cpu instead of gpu 0:cpu 1:gpu
    config = tf.ConfigProto(device_count={'GPU':1})
    with tf.Session(config=config) as sess:

        sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver()
        # to visualize using TensorBoard
        writer = tf.summary.FileWriter('./graphs', sess.graph)
        ckpt = tf.train.get_checkpoint_state(os.path.dirname('./checkpoints/'))

        # if that checkpoint exists, restore from checkpoint
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)

        for epoch in range(n_epochs):

            total_loss_likelihood = 0.0
            total_loss_divergence = 0.0
            total_loss_dis =0.0
            # Random shuffling
            np.random.shuffle(train_total_data)
            #train_data_ = train_total_data[:, :-mnist_data.NUM_LABELS]

            # Loop over all batches
            for i in range(total_batch):
                # Compute the offset of the current minibatch in the data.
                offset = (i * batch_size) % (n_samples)
                batch_xs_input = train_total_data[offset:(offset + batch_size), :-12]
                batch_cond_info = train_total_data[offset:(offset + batch_size), -12:]

                # update autoencoder parameters
                #z0 = np.random.randn(128, dim_z)
                #_, loss_divergence ,enc_out= sess.run((train_op_Enc, loss_encoder,encoder_output),feed_dict={x: batch_xs_input, cond_info: batch_cond_info})
                _,loss_likelihood,loss_divergence= sess.run((train_op_AE,loss_reconstr,loss_encoder), feed_dict={ x: batch_xs_input, cond_info:batch_cond_info,batchsize:batch_xs_input.shape[0]})


                # update discriminator
                for _ in range(n_critic):

                    z0 = np.random.normal(loc=0., scale=1, size=(batch_size, dim_z))
                    _, loss_dis = sess.run((train_op_Dis, loss_discriminator),
                                           feed_dict={ x: batch_xs_input, cond_info: batch_cond_info, z: z0})
                # _ = sess.run(clipped_weights)

                total_loss_likelihood = total_loss_likelihood + loss_likelihood
                total_loss_divergence = total_loss_divergence + loss_divergence
                total_loss_dis = total_loss_dis + loss_dis

            total_loss_likelihood = total_loss_likelihood / total_batch
            total_loss_divergence = total_loss_divergence / total_batch
            total_loss_dis = total_loss_dis / total_batch
            tot_loss = total_loss_divergence + total_loss_likelihood

            epoch_array[epoch] = epoch
            loss_array[epoch] = total_loss_likelihood
            # print cost every epoch
            print("epoch %d:  L_likelihood %03.3f L_divergence %03.3f L_dis %03.3f" % (epoch, total_loss_likelihood*4096, total_loss_divergence,total_loss_dis))

            # if minimum loss is updated or final epoch, plot results
            #if min_tot_loss > tot_loss or min_tot_mar_loss > total_loss_likelihood or epoch+1 == n_epochs:
            if epoch %10==0:
                saver.save(sess, './checkpoints/checkpoint', epoch)
                min_tot_loss = tot_loss
                min_tot_mar_loss = total_loss_likelihood
                # Plot for reproduce performance
                if PRR:
                    #z_PRR = sess.run(encoder_output,feed_dict={x: x_PRR, cond_info: x_PRR_info})
                    y_PRR = sess.run(decoder_output, feed_dict={x: x_PRR, cond_info:x_PRR_info})
                    y_PRR_img = y_PRR.reshape(PRR.n_tot_imgs, IMAGE_SIZE_MNIST, IMAGE_SIZE_MNIST)
                    t_z = np.random.normal(loc=0., scale=1, size=(batch_size, dim_z))
                    t_rand = np.random.randint(0, 10, size=[test_batch_size, 1])  # [0,10)
                    t_info = num_to_one_hot(t_rand)
                    #t_angle = np.random.randint(0, 360, size=[test_batch_size, 1])  # [0,360)
                    t_angle = np.random.uniform(size=[test_batch_size, 1])*360  # [0,360)
                    sin_angle = np.sin(t_angle / 180.0 * np.pi)
                    cos_angle = np.cos(t_angle / 180.0 * np.pi)
                    t_info = np.concatenate((t_info, sin_angle), axis=1)
                    t_info = np.concatenate((t_info, cos_angle), axis=1)
                    x_test = sess.run(test_op, feed_dict={z_test: t_z, test_cond_info: t_info})
                    x_test= x_test.reshape(test_batch_size, IMAGE_SIZE_MNIST,IMAGE_SIZE_MNIST)
                    if epoch%10==0 :
                        PRR.save_images(y_PRR_img, name="/PRR_epoch_%02d" % (epoch) + ".jpg")
                        PRR.save_images(x_test, name="/GER_epoch_%02d" % (epoch) + ".jpg")



        if PRR and save_result:
           test_image=np.zeros([test_smaple_size,IMAGE_SIZE_MNIST,IMAGE_SIZE_MNIST],dtype=np.float32)
           test_sample=sess.run(test_sample)
           test_batch = int(test_smaple_size/test_batch_size)
           for i in range(test_batch):
               # Compute the offset of the current minibatch in the data.
               offset = i * test_batch_size
               test_input = test_sample[offset:(offset + test_batch_size), :]
               test_input_info = test_info[offset:(offset + test_batch_size), :]
               x_test=sess.run(test_op,feed_dict={z_test:test_input,test_cond_info:test_input_info})
               test_image[offset:(offset + test_batch_size),:,:] = x_test.reshape(test_batch_size, IMAGE_SIZE_MNIST, IMAGE_SIZE_MNIST)
           PRR.save_images(test_image[0:128,:,:], name="/PRR_test"  + ".jpg")
           savename="./fakeimdb_loss_%03.2f" % (tot_loss) + ".mat"
           sio.savemat(savename,{'fakeimdb':test_image})




if __name__ == '__main__':

    # parse arguments
    args = parse_args()
    if args is None:
        exit()

    # main
    main(args)