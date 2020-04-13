from cifar import Cifar
from tqdm import tqdm
import tensorflow as tf
import numpy as np
import pretrained
import helper



n_classes = 10
learning_rate = 0.001
batch_size = 16
no_of_epochs = 1
image_size = 224
#In this we remove the last convolution layer'fc8' and use 'fc7' to compute the output and give that as the input to softmax function
fc6 = pretrained.fc6
weights = tf.Variable(tf.zeros([4096, n_classes]), name="output_weight")
bias = tf.Variable(tf.truncated_normal([n_classes]), name="output_bias")
fc7 = tf.nn.relu_layer(fc6, weights, bias)
model = tf.nn.softmax(fc7)

outputs = tf.placeholder(tf.int32, [None, n_classes])
centroids = tf.get_variable('c',shape=[n_classes],dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer(),trainable=False)
centroids_delta = tf.get_variable('centroidsUpdateTempVariable',shape=[n_classes],dtype=tf.float32,initializer=tf.zeros_initializer(),trainable=False)
centroids_batch = tf.gather(centroids,outputs)
cost = tf.nn.l2_loss(model - centroids_batch) / float(batch_size)
#cost = tf.losses.softmax_cross_entropy(outputs, model)+*cLoss
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
correct_pred = tf.equal(tf.argmax(model, 1), tf.argmax(outputs, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
cifar = Cifar(batch_size=batch_size)

init = tf.initialize_all_variables()
saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(init)
    run_options = tf.RunOptions(report_tensor_allocations_upon_oom = True)

    for epoch in range(no_of_epochs):
        for i in tqdm(range(cifar.no_of_batches),
                desc="Epoch {}".format(epoch),
                unit=" batch "):
            this_batch = cifar.batch(i)
            input_batch, out = helper.reshape_batch(this_batch, (image_size, image_size), n_classes)

            
            sess.run([optimizer],
                        feed_dict={
                            pretrained.x: input_batch,
                            outputs: out },
                        options=run_options)

        acc, loss = sess.run([accuracy, cost],
                       feed_dict={
                           pretrained.x: input_batch,
                           outputs: out },
                       options=run_options)

        print("Acc: {} Loss: {}".format(acc, loss))
        saver.save(sess, "saved_model/alexnet.ckpt")
