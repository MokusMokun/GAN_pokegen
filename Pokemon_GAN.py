# By @Mirope Yuhao Hu
# Part of the code is retrieved from: https://github.com/llSourcell/Pokemon_GAN

import os, cv2, random, time
import tensorflow as tf
import numpy as np
import scipy.misc
from tensorflow.examples.tutorials.mnist import input_data
from utils import *

slim = tf.contrib.slim

# the constants to 
HEIGHT, WIDTH, CHANNEL = 128, 128, 3 # The constraint of images
LEARNING_RATE = 2e-4
BATCH_SIZE = 64
EPOCH = 301
newPoke_path = './newPokemon' # The dir of storing generated pokemon images
data_path = 'Data' # The data dir, the details of data retrieving is in the report
logPath = "./tb_log" # The tensorboard summary are generated in this dir

# This is the leaky relu function
def lrelu(x, n, leak=0.2): 
    return tf.maximum(x, leak * x, name=n) 
 
def process_data():   
    current_dir = os.getcwd()
    pokemon_dir = os.path.join(current_dir, 'Data')
    images = []
    for each in os.listdir(pokemon_dir):
        images.append(os.path.join(pokemon_dir,each))
    # print images    
    all_images = tf.convert_to_tensor(images, dtype = tf.string)
    
    images_queue = tf.train.slice_input_producer(
                                        [all_images])
                                        
    content = tf.read_file(images_queue[0])
    image = tf.image.decode_jpeg(content, channels = CHANNEL)
    image.set_shape([HEIGHT,WIDTH,CHANNEL])
    image = tf.cast(image, tf.float32)
    image = image / 255.0
    
    iamges_batch = tf.train.shuffle_batch(
                                    [image], batch_size = BATCH_SIZE,
                                    num_threads = 4, capacity = 200 + 3* BATCH_SIZE,
                                    min_after_dequeue = 200)
    num_images = len(images)

    return iamges_batch, num_images

def generator(input, random_dim, is_train):
    c4, c8, c16, c32, c64 = 512, 256, 128, 64, 32 # filter sizes
    s4 = 4
    output_dim = CHANNEL  # RGB image

    with tf.variable_scope('gen', reuse=tf.AUTO_REUSE) as scope:

        w1 = tf.get_variable('w1', shape=[random_dim, s4 * s4 * c4], dtype=tf.float32,
                             initializer=tf.truncated_normal_initializer(stddev=0.02))
        b1 = tf.get_variable('b1', shape=[c4 * s4 * s4], dtype=tf.float32,
                             initializer=tf.constant_initializer(0.0))
        flat_conv1 = tf.add(tf.matmul(input, w1), b1, name='flat_conv1')
        # First set of layer
        conv1 = tf.reshape(flat_conv1, shape=[-1, s4, s4, c4], name='conv1')
        bn1 = tf.contrib.layers.batch_norm(conv1, is_training=is_train, epsilon=1e-5, decay = 0.9,  updates_collections=None, scope='bn1')
        act1 = tf.nn.relu(bn1, name='act1')
        # Second set of layer with a transpose layer.
        conv2 = tf.layers.conv2d_transpose(act1, c8, kernel_size=[5, 5], strides=[2, 2], padding="SAME",
                                           kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                           name='conv2')
        bn2 = tf.contrib.layers.batch_norm(conv2, is_training=is_train, epsilon=1e-5, decay = 0.9,  updates_collections=None, scope='bn2')
        act2 = tf.nn.relu(bn2, name='act2')
        # Third set of layer with a transpose layer
        conv3 = tf.layers.conv2d_transpose(act2, c16, kernel_size=[5, 5], strides=[2, 2], padding="SAME",
                                           kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                           name='conv3')
        bn3 = tf.contrib.layers.batch_norm(conv3, is_training=is_train, epsilon=1e-5, decay = 0.9,  updates_collections=None, scope='bn3')
        act3 = tf.nn.relu(bn3, name='act3')
        # Fourth set of layer with a transpose layer
        conv4 = tf.layers.conv2d_transpose(act3, c32, kernel_size=[5, 5], strides=[2, 2], padding="SAME",
                                           kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                           name='conv4')
        bn4 = tf.contrib.layers.batch_norm(conv4, is_training=is_train, epsilon=1e-5, decay = 0.9,  updates_collections=None, scope='bn4')
        act4 = tf.nn.relu(bn4, name='act4')
        # Fifth set of layer with a transpose layer
        conv5 = tf.layers.conv2d_transpose(act4, c64, kernel_size=[5, 5], strides=[2, 2], padding="SAME",
                                           kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                           name='conv5')
        bn5 = tf.contrib.layers.batch_norm(conv5, is_training=is_train, epsilon=1e-5, decay = 0.9,  updates_collections=None, scope='bn5')
        act5 = tf.nn.relu(bn5, name='act5')
        
        # Sixth set of layer
        conv6 = tf.layers.conv2d_transpose(act5, output_dim, kernel_size=[5, 5], strides=[2, 2], padding="SAME",
                                           kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                           name='conv6')
        act6 = tf.nn.tanh(conv6, name='act6')
        return act6


def discriminator(input, is_train, mnist=False):
    c2, c4, c8, c16 = 64, 128, 256, 512  # filter sizes

    with tf.variable_scope('dis', reuse=tf.AUTO_REUSE) as scope:

        # First set of layers with a convolution layer
        conv1 = tf.layers.conv2d(input, c2, kernel_size=[5, 5], strides=[2, 2], padding="SAME",
                                 kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                 name='conv1')
        bn1 = tf.contrib.layers.batch_norm(conv1, is_training = is_train, epsilon=1e-5, decay = 0.9,  updates_collections=None, scope = 'bn1')
        act1 = lrelu(conv1, n='act1')
        # Second set of layers with a convolution layer
        conv2 = tf.layers.conv2d(act1, c4, kernel_size=[5, 5], strides=[2, 2], padding="SAME",
                                 kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                 name='conv2')
        bn2 = tf.contrib.layers.batch_norm(conv2, is_training=is_train, epsilon=1e-5, decay = 0.9,  updates_collections=None, scope='bn2')
        act2 = lrelu(bn2, n='act2')
        # Third set of layers with a convolution layer
        conv3 = tf.layers.conv2d(act2, c8, kernel_size=[5, 5], strides=[2, 2], padding="SAME",
                                 kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                 name='conv3')
        bn3 = tf.contrib.layers.batch_norm(conv3, is_training=is_train, epsilon=1e-5, decay = 0.9,  updates_collections=None, scope='bn3')
        act3 = lrelu(bn3, n='act3')
        # Fourth set of layers with a convolution layer
        conv4 = tf.layers.conv2d(act3, c16, kernel_size=[5, 5], strides=[2, 2], padding="SAME",
                                 kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                 name='conv4')
        bn4 = tf.contrib.layers.batch_norm(conv4, is_training=is_train, epsilon=1e-5, decay = 0.9,  updates_collections=None, scope='bn4')
        act4 = lrelu(bn4, n='act4')

        # Testing the capability of discriminator using MNIST
        if mnist: 
            return act4

        # Fifth set of layer, readout layer
        dim = int(np.prod(act4.get_shape()[1:]))
        fc1 = tf.reshape(act4, shape=[-1, dim], name='fc1')
        w2 = tf.get_variable('w2', shape=[fc1.shape[-1], 1], dtype=tf.float32,
                             initializer=tf.truncated_normal_initializer(stddev=0.02))
        b2 = tf.get_variable('b2', shape=[1], dtype=tf.float32,
                             initializer=tf.constant_initializer(0.0))
        logits = tf.add(tf.matmul(fc1, w2), b2, name='logits')

        return logits


def train():
    random_dim = 100
    
    with tf.variable_scope('input'):
        #real and fake image placholders
        real_image = tf.placeholder(tf.float32, shape = [None, HEIGHT, WIDTH, CHANNEL], name='real_image')
        random_input = tf.placeholder(tf.float32, shape=[None, random_dim], name='rand_input')
        is_train = tf.placeholder(tf.bool, name='is_train')
    
    # wgan
    fake_image = generator(random_input, random_dim, is_train)
    
    real_result = discriminator(real_image, is_train)
    fake_result = discriminator(fake_image, is_train)

    real_predictions = tf.cast(real_result > 0, tf.float32)
    fake_predictions = tf.cast(fake_result < 0, tf.float32)
    num_predictions = 2.0*BATCH_SIZE
    num_correct = tf.reduce_sum(real_predictions) + tf.reduce_sum(fake_predictions)
    d_accuracy = tf.summary.scalar(name='d_accuracy', tensor=num_correct / num_predictions)
    
    d_loss = tf.reduce_mean(fake_result) - tf.reduce_mean(real_result)  # This optimizes the discriminator.
    d_summary = tf.summary.scalar(name='d_loss', tensor=d_loss)
    g_loss = -tf.reduce_mean(fake_result)  # This optimizes the generator.
    g_summary = tf.summary.scalar(name='g_loss', tensor=g_loss)

    t_vars = tf.trainable_variables()
    d_vars = [var for var in t_vars if 'dis' in var.name]
    g_vars = [var for var in t_vars if 'gen' in var.name]
    trainer_d = tf.train.RMSPropOptimizer(learning_rate=LEARNING_RATE).minimize(d_loss, var_list=d_vars)
    trainer_g = tf.train.RMSPropOptimizer(learning_rate=LEARNING_RATE).minimize(g_loss, var_list=g_vars)
    # clip discriminator weights
    d_clip = [v.assign(tf.clip_by_value(v, -0.01, 0.01)) for v in d_vars]

    
    batch_size = BATCH_SIZE
    image_batch, samples_num = process_data()
    
    batch_num = int(samples_num / batch_size)
    total_batch = 0
    sess = tf.Session()
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    # continue training
    # saver.restore(sess, './tmp/model.ckpt')
    save_path = saver.save(sess, "./tmp/model.ckpt")
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    tbWriter = tf.summary.FileWriter(logPath, sess.graph)

    start_time = time.time()

    print('total training sample num:%d' % samples_num)
    print('batch size: %d, batch num per epoch: %d, epoch num: %d' % (batch_size, batch_num, EPOCH))
    print('start training...')
    for i in range(EPOCH):

        for j in range(batch_num):
            d_iters = 5
            g_iters = 1

            batch_start = time.time()
            train_noise = np.random.uniform(-1.0, 1.0, size=[batch_size, random_dim]).astype(np.float32)
            for k in range(d_iters):
                train_image = sess.run(image_batch)
                #wgan clip weights
                sess.run(d_clip)
                
                # Update the discriminator
                _, dLoss, d_acc = sess.run([trainer_d, d_summary, d_accuracy],
                                    feed_dict={random_input: train_noise, real_image: train_image, is_train: True})
                tbWriter.add_summary(dLoss, i)
                tbWriter.add_summary(d_acc, i)

            # Update the generator
            for k in range(g_iters):
                _, gLoss = sess.run([trainer_g, g_summary],
                                    feed_dict={random_input: train_noise, is_train: True})
                tbWriter.add_summary(gLoss, i)

            time_dur = time.time() - start_time
            time_batch = time.time() - batch_start
            print('training on EPOCH {}/{}, batch {}/{}, time elapsed: {:.3f} sec, batch time {:.3f}'.\
            format(i, EPOCH, j, batch_num, time_dur, time_batch), end='\r')
            
        # save check point every 50 epoches
        if i%50 == 0:
            if not os.path.exists('./model/'):
                os.makedirs('./model/')
            saver.save(sess, './model/' + str(i))  
        if i%5 == 0:
            # save images
            if not os.path.exists(newPoke_path):
                os.makedirs(newPoke_path)
            sample_noise = np.random.uniform(-1.0, 1.0, size=[batch_size, random_dim]).astype(np.float32)
            imgtest = sess.run(fake_image, feed_dict={random_input: sample_noise, is_train: False})
            save_images(imgtest, [8,8] ,newPoke_path + '/epoch' + str(i) + '.jpg')
    
    time_dur = time.time() - start_time
    print("\nTraining ended for {} EPOCHes with mini batch of {}, {:.3f} secs".format(EPOCH,BATCH_SIZE,time_dur))

    coord.request_stop()
    coord.join(threads)

def weight_variable(shape, name=None):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=name)

def bias_variable(shape, name=None):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=name)

def mnist():
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    sess = tf.InteractiveSession()

    # place holders for MNIST input data
    x = tf.placeholder(tf.float32, shape=[None, 784], name="x")
    y_ = tf.placeholder(tf.float32, [None, 10], name="y_")

    # change the MNIST input data from a list of values to a 28 pixel X 28 pixel X 1 grayscale value cube
    #    which the Convolution NN can use.
    input_image_shape = 28
    input_image = tf.reshape(x, [-1,input_image_shape,input_image_shape,1], name="x_image")

    # run through the discriminator nn
    mnist_result = discriminator(input_image, False, True) # is_train is false

    fully1 = 2
    fully2 = 512
    Readout_size = 1024
    # Fully Connected Layer
    W_fc1 = weight_variable([fully1 * fully1 * fully2, Readout_size], name="weight")
    b_fc1 = bias_variable([Readout_size], name="bias")
    #   Connect output of pooling layer 2 as input to full connected layer
    h_pool2_flat = tf.reshape(mnist_result, [-1, fully1 * fully1 * fully2])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1, name="relu")

    # dropout some neurons to reduce overfitting
    keep_prob = tf.placeholder(tf.float32, name="keep_prob")  # get dropout probability as a training input.
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # Readout layer
    W_fc2 = weight_variable([Readout_size, 10], name="weight")
    b_fc2 = bias_variable([10], name="bias")

    # Define model
    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    # Loss measurement
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=y_conv, labels=y_))

    # loss optimization
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    # What is correct
    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
    # How accurate is it?
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    sess.run(tf.global_variables_initializer())

    #  define number of steps and how often we display progress
    num_steps = 2000
    display_every = 100

    # Start timer
    start_time = time.time()
    end_time = time.time()
    for i in range(num_steps):
        batch = mnist.train.next_batch(50)
        sess.run([train_step], feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})


        # Periodic status display
        if i%display_every == 0:
            train_accuracy = accuracy.eval(feed_dict={
                x:batch[0], y_: batch[1], keep_prob: 1.0})
            end_time = time.time()
            print("step {0}, elapsed time {1:.2f} seconds, training accuracy {2:.3f}%".format(i, \
            end_time-start_time, train_accuracy*100.0))

    # Display summary 
    #     Time to train
    end_time = time.time()
    print("Total training time for {0} batches: {1:.2f} seconds".format(i+1, end_time-start_time))
    #     Accuracy on test data
    print("Test accuracy {0:.3f}%".format(accuracy.eval(feed_dict={
        x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})*100.0))

if __name__ == "__main__":
    train() # This fuction execute the training process
    mnist() # This function execute the discriminator capability testing process

