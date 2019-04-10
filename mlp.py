import tensorflow as tf
from data_helper import *
import sys
import datetime

# Data processing
trainData, validData, testData, trainTarget, validTarget, testTarget = loadData()
trainData, validData, testData = flattenData(trainData), flattenData(validData), flattenData(testData)
trainTarget, validTarget, testTarget = convertOneHot(trainTarget, validTarget, testTarget)
# print(len(trainData) + len(validData) + len(testData))
# sys.exit(0)

# Parameters
learning_rate = 0.0005
n_epochs = 50
batch_size = 64
L2_norm = 0.1

# Network Parameters
n_hidden = 256 # 1st layer number of neurons
n_input = 784 # MNIST data input (img shape: 28*28)
n_classes = 10 # MNIST total classes (0-9 digits)

# tf Graph input
X = tf.placeholder("float", [None, n_input])
Y = tf.placeholder("float", [None, n_classes])

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden])),
    'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}


# Create model
def multilayer_perceptron(x):
    # Hidden fully connected layer with 256 neurons
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])

    # Output fully connected layer with a neuron for each class
    out_layer = tf.matmul(layer_1, weights['out']) + biases['out']
    return out_layer


# Construct model
logits = multilayer_perceptron(X)
prediction = tf.nn.softmax(logits)

# Define loss and optimizer
regularization = tf.nn.l2_loss(weights['h1']) + tf.nn.l2_loss(weights['out'])
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y)) + \
                L2_norm * regularization
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# Evaluate model
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()

with tf.Session() as sess:
    # writer = tf.summary.FileWriter("mlp_output", sess.graph)
    sess.run(init)

    # divide mini-batches
    data_size = trainData.shape[0]
    num_batches_per_epoch = data_size // batch_size
    if data_size % batch_size: num_batches_per_epoch += 1

    print(datetime.datetime.now())
    # mini-batch
    for i in range(n_epochs):
        # shuffle data in each epoch
        trainData, trainTarget = shuffle(trainData, trainTarget)

        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            feed_dict = {
                X: trainData[start_index:end_index],
                Y: trainTarget[start_index:end_index]
            }
            sess.run(train_op, feed_dict=feed_dict)

        # trainloss, trainacc = sess.run([loss_op, accuracy], feed_dict={
        #     X: trainData,
        #     Y: trainTarget
        # })
        #
        # validloss, validacc = sess.run([loss_op, accuracy], feed_dict={
        #     X: validData,
        #     Y: validTarget
        # })
        # testloss, testacc = sess.run([loss_op, accuracy], feed_dict={
        #     X: testData,
        #     Y: testTarget
        # })
        #
        # print("epoch: {}, trainloss: {}, validloss: {}, testloss: {}, trainacc: {}, validacc: {}, testacc: {}" \
        #       .format(i, trainloss, validloss, testloss, trainacc, validacc, testacc))
    # writer.close()
    print(datetime.datetime.now())
