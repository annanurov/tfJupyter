#from __future__ import absolute_import
#from __future__ import division
#from __future__ import print_function

#import os
#import urllib

import numpy as np
import tensorflow as tf



from  os import environ
#os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
environ['TF_CPP_MIN_LOG_LEVEL']='2'

#from .
import getImage


# associate the "label" and "image" objects with the corresponding features read from
# a single example in the training data file

getImage.height = 32 #hieght and wigth of imported images.
getImage.width = 32

getImage.nClass = 33 #2

height = getImage.height
width = getImage.width
nClass = getImage.nClass

TRAINfolderName = "data/KHimages/train/"
#fileName = "16/resized1A-16X0.png"
# interactive session allows inteleaving of building and running steps
sess = tf.InteractiveSession()

#label, image = getImage.getImage("data/train-00000-of-00001")
#label, image = getImage.getImage("data/train")
#label, image = getImage.getImage(TRAINfolderName + fileName)
label, image = getImage.getImage(TRAINfolderName  + "train-00000-of-00001")
#print (label)

print ("OK\n")

# and similarly for the validation data

TESTfolderName = "data/KHimages/validate/"
#vlabel, vimage = getImage.getImage("data/validation-00000-of-00001")
vlabel, vimage = getImage.getImage(TESTfolderName + "/validation-00000-of-00001")

# associate the "label_batch" and "image_batch" objects with a randomly selected batch---
# of labels and images respectively
imageBatch, labelBatch = tf.train.shuffle_batch(
    [image, label], batch_size=100,
    capacity=2000,
    min_after_dequeue=1000)
    

# and similarly for the validation data
vimageBatch, vlabelBatch = tf.train.shuffle_batch(
    [vimage, vlabel], batch_size=100,
    capacity=2000,
    min_after_dequeue=1000)

print ("OK2\n")


# x is the input array, which will contain the data from an image
# this creates a placeholder for x, to be populated later
x = tf.placeholder(tf.float32, [None, getImage.width*getImage.height])
# similarly, we have a placeholder for true outputs (obtained from labels)
y_ = tf.placeholder(tf.float32, [None, getImage.nClass])



simpleModel = 1

if simpleModel:
  # run simple model y=Wx+b given in TensorFlow "MNIST" tutorial

  print "Running Simple Model y=Wx+b"

  # initialise weights and biases to zero
  # W maps input to output so is of size: (number of pixels) * (Number of Classes)
  W = tf.Variable(tf.zeros([width*height, nClass]))
  # b is vector which has a size corresponding to number of classes
  b = tf.Variable(tf.zeros([nClass]))

  # define output calc (for each class) y = softmax(Wx+b)
  # softmax gives probability distribution across all classes
  y = tf.nn.softmax(tf.matmul(x, W) + b)



else:
    ####Run more complex things here
    print ("Run more complex things here\n")

# measure of error of our model
# this needs to be minimised by adjusting W and b
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

# define training step which minimises cross entropy
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# argmax gives index of highest entry in vector (1st axis of 1D tensor)
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
# get mean of all entries in correct prediction, the higher the better
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# initialize the variables
sess.run(tf.global_variables_initializer())

# start the threads used for reading files
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess,coord=coord)

# start training
nSteps=10
for i in range(nSteps):
    print "OK -- " + str(i) + "\n"
    batch_xs, batch_ys = sess.run([imageBatch, labelBatch])
    continue
    # run the training step with feed of images
    if simpleModel:
      train_step.run(feed_dict={x: batch_xs, y_: batch_ys})
    else:
      train_step.run(feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.5})


    if (i+1)%100 == 0: # then perform validation

      # get a validation batch
      vbatch_xs, vbatch_ys = sess.run([vimageBatch, vlabelBatch])
      if simpleModel:
        train_accuracy = accuracy.eval(feed_dict={
          x:vbatch_xs, y_: vbatch_ys})
      else:
        train_accuracy = accuracy.eval(feed_dict={
          x:vbatch_xs, y_: vbatch_ys, keep_prob: 1.0})
      print("step %d, training accuracy %g"%(i+1, train_accuracy))
print "OK3\n"
"""


# finalise
coord.request_stop()
coord.join(threads)
"""
