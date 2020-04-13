from cifar import Cifar
from tqdm import tqdm
import tensorflow as tf
import numpy as np
import pretrained
import helper



n_classes = 10 #This is the number of classes in the new dataset-CIFAR-10
batch_size = 16
no_of_epochs = 2
image_size = 224 #This is the image size of the images in the pretrained weights
learning_rate = 0.01
decay_rate = learning_rate #We change this learning rate with the epochs

fc7 = pretrained.fc7
weights = tf.Variable(tf.zeros([4096, n_classes]), name="output_weight")
bias = tf.Variable(tf.truncated_normal([n_classes]), name="output_bias")
fc8 = tf.matmul(fc7, weights) + bias
model = tf.nn.softmax(fc8)

outputs = tf.placeholder(tf.float32, [None, n_classes])
cost = tf.losses.softmax_cross_entropy(outputs, model)
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=decay_rate).minimize(cost)
correct_pred = tf.equal(tf.argmax(model, 1), tf.argmax(outputs, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
cifar = Cifar(batch_size=batch_size)

init = tf.initialize_all_variables()
saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(init)
    run_options = tf.RunOptions(report_tensor_allocations_upon_oom = True)

    for epoch in range(no_of_epochs):
        #This part of the code includes a dynamic learning rate which varies with the epochs
        decay_rate = learning_rate/(epoch+1)
        print(f"The learning rate used in this epoch was {decay_rate}")
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

        