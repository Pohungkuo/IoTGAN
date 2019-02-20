import tensorflow as tf
import numpy as np
import png
import random
import os
import code

import sys
import time
from subprocess import call

#import IoTmain_colab_dis

class GAN:

    def __init__(self, task_number_1):
        #cluster = tf.train.ClusterSpec({"PS":["localhost:2222"], "DSGraph":["localhost:2223", "localhost:2224"]})
        cluster = tf.train.ClusterSpec({"PS":["192.168.0.101:2222"], "DSGraph":["192.168.0.101:2223", "192.168.0.103:2224"]})
        server = tf.train.Server(cluster, job_name="DSGraph", task_index=task_number_1)

        with tf.device(tf.train.replica_device_setter(worker_device="/job:DSGraph/task:{}".format(task_number_1), cluster=cluster)):

            self.is_training = tf.placeholder(tf.bool, name='is_training')

            self.g_x, self.g_y, self.g_y_logits = self.build_generator()

            with tf.variable_scope('discriminator') as scope:

                self.d_x = tf.placeholder(tf.float32, shape=[None, 784])
                self.d_y_ = tf.placeholder(tf.float32, shape=[None, 1])
                self.d_keep_prob = tf.placeholder(tf.float32, name='d_keep_prob')

                self.d_y, self.d_y_logit = self.build_discriminator(self.d_x, self.d_keep_prob)

                scope.reuse_variables()
                self.g_d_y, self.g_d_y_logit = self.build_discriminator(self.g_y, self.d_keep_prob)

            vars = tf.trainable_variables()
            # build loss function for discriminator
            d_loss_real = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d_y_logit, labels=tf.ones_like(self.d_y_logit))
            d_loss_fake = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.g_d_y_logit, labels=tf.zeros_like(self.g_d_y_logit))
            self.d_loss = d_loss_real + d_loss_fake
            d_training_vars = [v for v in vars if v.name.startswith('discriminator/')]
            self.d_optimizer = tf.train.AdamOptimizer(0.0002, beta1=0.5).minimize(self.d_loss, var_list=d_training_vars)

            self.d_accuracy = tf.reduce_sum(tf.cast(tf.equal(tf.round(self.d_y_logit), tf.round(self.d_y_)), tf.float32))


            # build loss function for training the generator
            self.g_d_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.g_d_y_logit, labels=tf.ones_like(self.g_d_y_logit))
            g_training_vars = [v for v in vars if v.name.startswith('generator/')]
            self.g_d_optimizer = tf.train.AdamOptimizer(0.0002, beta1=0.5).minimize(self.g_d_loss, var_list=g_training_vars)


    def restore_session(self, path):
        saver = tf.train.Saver()
        sess = tf.Session()
        saver.restore(sess, tf.train.latest_checkpoint(path))
        return sess

    def train_digit(self, mnist, digit, path, path_GAN2, path_GAN3, task_number_1):
        if (task_number_1 == 0):
            sess = tf.Session("grpc://192.168.0.101:2222", config=tf.ConfigProto(log_device_placement=True))
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver(sharded=True)

            #tf.train.import_meta_graph('./IoTGAN/digit-1/restore/model-2600.meta')
            #saver.restore(sess, tf.train.latest_checkpoint('./IoTGAN/digit-1/restore'))

            # Get all the training '1' digits for our "real" data
            train_digits_of_interest = []
            #train_digits_of_interest_2 = []

            for image, label in zip(mnist.train.images, mnist.train.labels):
                #image_index = np.where(mnist.train.images==image)
                #if label[0]:
                #    if image_index[0][0]%10!=0:
                #        train_digits_of_interest.append(image)
                #if label[2]:
                #    if image_index[0][0]%10==0:
                #        train_digits_of_interest.append(image)
                if label[digit]:
                    train_digits_of_interest.append(image)
                #if label[2]:
                #    train_digits_of_interest_2.append(image)


            test_digits_of_interest = []
            for image, label in zip(mnist.test.images, mnist.test.labels):
                if label[digit]:
                    test_digits_of_interest.append(image)

            random.seed(12345)
            random.shuffle(train_digits_of_interest)
            random.shuffle(test_digits_of_interest)
            batch_size = 32
            for step in range(20000):

                batch_index = step * batch_size % len(train_digits_of_interest)
                batch_index = min(batch_index, len(train_digits_of_interest) - batch_size)
                batch = train_digits_of_interest[batch_index:(batch_index + batch_size)]
                batch_print = np.reshape(batch, (32*28, 28)) * 255.0
                #png.save_png('%s/1stGAN_InputArray_digit-step-%06d.png' % (os.path.dirname(path), step), batch_print)

                #
                # Train the discriminator
                _, discriminator_loss = sess.run([self.d_optimizer, self.d_loss], feed_dict={self.is_training: True, self.d_x: batch, self.g_x: np.random.normal(size=(32,32)), self.d_keep_prob: 0.5})
                #
                # Train the generator
                z = np.random.normal(size=(32,32))
                _, generator_loss = sess.run([self.g_d_optimizer, self.g_d_loss], feed_dict={self.is_training: True, self.g_x: z, self.d_keep_prob: 1.0})

                if step % 1 == 0:
                    print "Digit %d Step %d Eval: %f %f" % (digit, step, discriminator_loss[0], generator_loss[0])
                if step % 2 == 0:
                    result = self.eval_generator(sess, 32)
                    image = np.reshape(result, (32*28, 28)) * 255.0
                    png.save_png('%s/digit-step-%06d.png' % (os.path.dirname(path), step), image)
                    saver.save(sess, path, step)

                    total_accuracy = 0
                    total_samples = 0
                    num_batches = 5
                    for i in xrange(num_batches):
                        fake_samples = [(x, 0.0) for x in self.eval_generator(sess, 32)]
                        real_samples = [(x, 1.0) for x in random.sample(test_digits_of_interest, 32)]
                        samples = fake_samples + real_samples
                        random.shuffle(samples)
                        xs, ys = zip(*samples)
                        xs = np.asarray(xs)
                        ys = np.asarray(ys)
                        ys = np.reshape(ys, (64, 1))
                        accuracy = sess.run([self.d_accuracy], feed_dict={self.is_training: False, self.d_x: xs, self.d_y_: ys, self.d_keep_prob: 1.0})
                        total_accuracy += accuracy[0]
                        total_samples += len(samples)
                    print("Discriminator eval accuracy %f%%" % (total_accuracy * 100.0 / total_samples))
                    if (total_accuracy * 100.0 / total_samples) != 50.0:
                        if (total_accuracy * 100.0 / total_samples) >= 15.0:
                            break

            saver.save(sess, path, step)

            cmd = "scp -r ./IoTGAN_dis/transfer pi@192.168.0.103:./mnist-gan/"
            call(cmd.split(" "))
            print("weight files updated to other pi")

            print(batch_index)
            sess.close()

        if (task_number_1==1):

            sess2 = tf.Session("grpc://192.168.0.101:2222", config=tf.ConfigProto(log_device_placement=True))
            sess2.run(tf.global_variables_initializer())
            saver2 = tf.train.Saver()
            print("come here1")
            sess = tf.Session("grpc://192.168.0.101:2222", config=tf.ConfigProto(log_device_placement=True))
            print("come here2")

            new_saver=tf.train.import_meta_graph('./IoTGAN_dis/digit-2/model-2200.meta')
            print("come here3")

            new_saver.restore(sess, tf.train.latest_checkpoint('./IoTGAN_dis/digit-2/'))
            print("come here4")

            train_digits_of_interest_2 = []
            print("come here5")

            for image_2, label_2 in zip(mnist.train.images, mnist.train.labels):
                #image_index = np.where(mnist.train.images==image)
                #if label[0]:
                #    if image_index[0][0]%10!=0:
                #        train_digits_of_interest.append(image)
                #if label[2]:
                #    if image_index[0][0]%10==0:
                #        train_digits_of_interest.append(image)
                print("come here6")

                if label_2[2]:
                    train_digits_of_interest_2.append(image_2)
                    result_single_GAN1 = self.eval_generator(sess, 32)
                    result_single_GAN1 = np.reshape(result_single_GAN1[0:1], (784,))

                    #print("shape_train_digits_of_interest_2", train_digits_of_interest_2.shape)
                    #print("shape_image_2", image_2.shape)
                    #print("shape_result_single_GAN1", result_single_GAN1.shape)
                    #png.save_png('%s/GAN1-generated-step-%06d.png' % (os.path.dirname(path), step), np.reshape(result_single_GAN1, (32*28, 28)) * 255.0)
                    train_digits_of_interest_2.append(result_single_GAN1)
                    print("come here7")

            test_digits_of_interest = []
            for image, label in zip(mnist.test.images, mnist.test.labels):
                if label[digit]:
                    test_digits_of_interest.append(image)

            random.seed(12345)
            print ("come here")
            random.shuffle(train_digits_of_interest_2)
            random.shuffle(test_digits_of_interest)
            batch_size = 32
            batch_index = 0

            for step in range(20000):
                #random.shuffle(train_digits_of_interest_2)

                batch_index = step * batch_size % len(train_digits_of_interest_2)
                batch_index = min(batch_index, len(train_digits_of_interest_2) - batch_size)
                batch = train_digits_of_interest_2[batch_index:(batch_index + batch_size)]
                batch_print = np.reshape(batch, (32*28, 28)) * 255.0
                #png.save_png('%s/2ndGAN_InputArray_digit-step-%06d.png' % (os.path.dirname(path), step), batch_print)

                #
                # Train the discriminator
                _, discriminator_loss2 = sess2.run([self.d_optimizer, self.d_loss], feed_dict={self.is_training: True, self.d_x: batch, self.g_x: np.random.normal(size=(32,32)), self.d_keep_prob: 0.5})

                #
                # Train the generator
                z = np.random.normal(size=(32,32))
                _, generator_loss2 = sess2.run([self.g_d_optimizer, self.g_d_loss], feed_dict={self.is_training: True, self.g_x: z, self.d_keep_prob: 1.0})

                if step % 100 == 0:
                    print "2nd GAN Digit %d Step %d Eval: %f %f" % (digit, step, discriminator_loss2[0], generator_loss2[0])

                if step % 200 == 0:
                    result = self.eval_generator(sess2, 32)
                    image = np.reshape(result, (32*28, 28)) * 255.0
                    png.save_png('%s/GAN2-InputTrainArray-step-%06d.png' % (os.path.dirname(path), step), np.reshape(batch, (32*28, 28)) * 255.0)

                    png.save_png('%s/2ndGAN-digit-step-%06d.png' % (os.path.dirname(path), step), image)
                    saver2.save(sess2, path_GAN2, step)

                    total_accuracy = 0
                    total_samples = 0
                    num_batches = 5
                    for i in xrange(num_batches):
                        fake_samples = [(x, 0.0) for x in self.eval_generator(sess2, 32)]
                        real_samples = [(x, 1.0) for x in random.sample(test_digits_of_interest, 32)]
                        samples = fake_samples + real_samples
                        random.shuffle(samples)
                        xs, ys = zip(*samples)
                        xs = np.asarray(xs)
                        ys = np.asarray(ys)
                        ys = np.reshape(ys, (64, 1))
                        accuracy = sess2.run([self.d_accuracy], feed_dict={self.is_training: False, self.d_x: xs, self.d_y_: ys, self.d_keep_prob: 1.0})
                        total_accuracy += accuracy[0]
                        total_samples += len(samples)
                    print("2nd GAN Discriminator eval accuracy %f%%" % (total_accuracy * 100.0 / total_samples))

                    #if step % 3 == 0:
                    #    result_single_GAN1 = self.eval_generator(sess, 32)
                    #    png.save_png('%s/GAN1-generated-step-%06d.png' % (os.path.dirname(path), step), np.reshape(result_single_GAN1, (32*28, 28)) * 255.0)

                    #    train_digits_of_interest_2.append(result_single_GAN1)

                    if (total_accuracy * 100.0 / total_samples) != 50.0:
                        if (total_accuracy * 100.0 / total_samples) >= 70.0:
                            break
            saver.save(sess2, path_GAN2, step)
            print(batch_index)
            print("GAN3 Start!GAN3 Start!GAN3 Start!GAN3 Start!GAN3 Start!GAN3 Start!")
            #counter = 0
            #j = step
            #print (j)
            #batch2 [:] = []
            #batch3 = []
            #step = step + 1
            #batch_index = batch_index + 1
            #print ("Trained 1 step, Trained 1 step, Trained 1 step, Trained 1 step,")




        while (1):
            time.sleep(10)
            #print ('1st GAN done')



    def input_data_filter(self, batch_index, step, batch_size_2, train_digits_of_interest, result_single, path, counter, batch2, batch3, sess):
        batch_index = step * batch_size_2 % len(train_digits_of_interest)
        batch_index = min(batch_index, len(train_digits_of_interest) - batch_size_2)
        batch = train_digits_of_interest[batch_index:(batch_index + batch_size_2)]
        #print("step_filter:", step)

        batch_reshape = np.reshape(batch, (32, 784)) * 255.0

        discriminator_loss = sess.run([self.d_loss], feed_dict={self.is_training: False, self.d_x: result_single, self.g_y: batch_reshape, self.d_keep_prob: 0.5})
        #print ("GAN Compare loss=" , discriminator_loss)
        #print("batch_index:", batch_index)
        #print(np.asarray(discriminator_loss).shape)

        for i in range (batch_size_2):
            if (np.asarray(discriminator_loss)[0][i][0]) > 2200:

                counter = counter + 1
                if (counter < 33):
                    batch2.append(train_digits_of_interest[batch_index + i])
                    print ("counter:", counter)
                    #print ("step_filter:", step)
                if (counter == 33):
                    batch3 = batch2
                    print ("batche3ready:")

        step = step + 1
        return batch3, counter, batch_index, step

    def input_data_filter_2(self, batch_index, step, batch_size_2, train_digits_of_interest, result_single_GAN1, result_single_GAN2, path, counter, batch2, batch3, sess, sess2):
        batch_index = step * batch_size_2 % len(train_digits_of_interest)
        batch_index = min(batch_index, len(train_digits_of_interest) - batch_size_2)
        batch = train_digits_of_interest[batch_index:(batch_index + batch_size_2)]
        #print("step_filter:", step)

        batch_reshape = np.reshape(batch, (32, 784)) * 255.0

        discriminator_loss_GAN1 = sess.run([self.d_loss], feed_dict={self.is_training: False, self.d_x: result_single_GAN1, self.g_y: batch_reshape, self.d_keep_prob: 0.5})
        #print ("GAN1 Compare loss=" , discriminator_loss_GAN1[0])
        #print("batch_index:", batch_index)
        #print(np.asarray(discriminator_loss).shape)

        discriminator_loss_GAN2 = sess2.run([self.d_loss], feed_dict={self.is_training: False, self.d_x: result_single_GAN2, self.g_y: batch_reshape, self.d_keep_prob: 0.5})
        #print ("GAN2 Compare loss=" , discriminator_loss_GAN2[0])
        #print("batch_index:", batch_index)


        for i in range (batch_size_2):
            if (np.asarray(discriminator_loss_GAN1)[0][i][0]) > 2200:
                if (np.asarray(discriminator_loss_GAN2)[0][i][0]) < 0.8:

                    counter = counter + 1
                    if (counter < 33):
                        batch2.append(train_digits_of_interest[batch_index + i])
                        print ("counter:", counter)
                        #print ("step_filter:", step)
                    if (counter == 33):
                        batch3 = batch2
                        print ("batche3ready:")

        step = step + 1
        return batch3, counter, batch_index, step

    def eval_generator(self, sess, n_samples=1):
        result = sess.run([self.g_y], {self.is_training: False, self.g_x: np.random.normal(size=(n_samples,32))})
        return result[0]

    def leakyrelu(self, x):
        return tf.maximum(0.01*x,x)
        #return tf.nn.relu(x)

    def batch_norm(self, x):
        return tf.contrib.layers.batch_norm(x, decay=0.9, scale=True, is_training=self.is_training, updates_collections=None)

    def build_generator(self):
        with tf.variable_scope('generator') as scope:
            g_x = tf.placeholder(tf.float32, shape=[None, 32], name='input')

            with tf.variable_scope("fc1"):
                g_w1 = tf.get_variable("g_w1", shape=[32, 1024], initializer=tf.contrib.layers.xavier_initializer())
                g_b1 = tf.get_variable("g_b1", initializer=tf.zeros([1024]))
                g_h1 = self.leakyrelu(self.batch_norm(tf.matmul(g_x, g_w1) + g_b1))

            with tf.variable_scope("fc2"):
                g_w2 = tf.get_variable("g_w2", shape=[1024, 7*7*64], initializer=tf.contrib.layers.xavier_initializer())
                g_b2 = tf.get_variable("g_b2", initializer=tf.zeros([7*7*64]))
                g_h2 = self.leakyrelu(self.batch_norm(tf.matmul(g_h1, g_w2) + g_b2))
                g_h2_reshaped = tf.reshape(g_h2, [-1, 7, 7, 64])

            with tf.variable_scope("conv3"):
                g_w3 = tf.get_variable("g_w3", shape=[5, 5, 32, 64], initializer=tf.contrib.layers.xavier_initializer())
                g_b3 = tf.get_variable("g_b3", initializer=tf.zeros([32]))
                g_deconv3 = tf.nn.conv2d_transpose(g_h2_reshaped, g_w3, output_shape=[32, 14, 14, 32], strides=[1, 2, 2, 1])
                g_h3 = self.leakyrelu(self.batch_norm(g_deconv3 + g_b3))

            with tf.variable_scope("conv4"):
                g_w4 = tf.get_variable("g_w4", shape=[5, 5, 1, 32], initializer=tf.contrib.layers.xavier_initializer())
                g_b4 = tf.get_variable("g_b4", initializer=tf.zeros([1]))
                g_deconv4 = tf.nn.conv2d_transpose(g_h3, g_w4, output_shape=[32, 28, 28, 1], strides=[1, 2, 2, 1])

            g_y_logits = tf.reshape(g_deconv4 + g_b4, [-1, 784])
            g_y = tf.nn.sigmoid(g_y_logits)
        return g_x, g_y, g_y_logits

    def build_discriminator(self, x, keep_prob):
        def weight_variable(shape):
          return tf.get_variable('weights', shape, initializer=tf.contrib.layers.xavier_initializer())

        def bias_variable(shape):
          return tf.get_variable('biases', shape, initializer=tf.constant_initializer(0.0))

        def conv2d(x, W):
          return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

        def max_pool_2x2(x):
          return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                                strides=[1, 2, 2, 1], padding='SAME')

        with tf.variable_scope("input"):
            d_x_image = tf.reshape(x, [-1,28,28,1])

        with tf.variable_scope("conv1"):
            d_W_conv1 = weight_variable([5, 5, 1, 32])
            d_b_conv1 = bias_variable([32])

            d_h_conv1 = self.leakyrelu(self.batch_norm(conv2d(d_x_image, d_W_conv1) + d_b_conv1))
            d_h_pool1 = max_pool_2x2(d_h_conv1)

        with tf.variable_scope("conv2"):
            d_W_conv2 = weight_variable([5, 5, 32, 64])
            d_b_conv2 = bias_variable([64])

            d_h_conv2 = self.leakyrelu(self.batch_norm(conv2d(d_h_pool1, d_W_conv2) + d_b_conv2))
            d_h_pool2 = max_pool_2x2(d_h_conv2)

        with tf.variable_scope("fc1"):
            d_W_fc1 = weight_variable([7 * 7 * 64, 1024])
            d_b_fc1 = bias_variable([1024])

            d_h_pool2_flat = tf.reshape(d_h_pool2, [-1, 7*7*64])
            d_h_fc1 = self.leakyrelu(self.batch_norm(tf.matmul(d_h_pool2_flat, d_W_fc1) + d_b_fc1))

            d_h_fc1_drop = tf.nn.dropout(d_h_fc1, keep_prob)

        with tf.variable_scope("fc2"):
            d_W_fc2 = weight_variable([1024, 1])
            d_b_fc2 = bias_variable([1])

        d_y_logit = tf.matmul(d_h_fc1_drop, d_W_fc2) + d_b_fc2
        d_y = tf.sigmoid(d_y_logit)

        return d_y, d_y_logit
