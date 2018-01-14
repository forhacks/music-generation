import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
event_len = 40
pitch_num = 120
data_size = pitch_num * 2 + 1
hidden_size = 500
epochs = 1000
num_layers = 3
batch_size = 30

print("Making Graph")
x = tf.placeholder("float", [None, event_len - 1, data_size])
y = tf.placeholder("float", [None, data_size])

layers = []
for l in range(num_layers):
    layers.append(rnn.BasicLSTMCell(hidden_size))
cell = rnn.MultiRNNCell(layers)

print("Feeding Inputs Into LSTM Layers")
timesteps = tf.unstack(x, event_len - 1, 1)
lstm_outputs, states = rnn.static_rnn(cell, timesteps, dtype=tf.float32)

w = tf.Variable(tf.truncated_normal([hidden_size, data_size], stddev=0.1))
b = tf.Variable(tf.truncated_normal([data_size], stddev=0.1))

print("Feeding Values Into Final Layer")
out = tf.matmul(lstm_outputs[-1], w) + b
loss = tf.reduce_sum(tf.losses.softmax_cross_entropy(y, out))

print('Computing Gradients')
train = tf.train.AdagradOptimizer(0.001).minimize(loss)
init = tf.global_variables_initializer()


def to_roll_matrix(data):
    inputs = []
    outputs = []
    for track in data:
        t = np.zeros((event_len, data_size))
        index = 0
        for event in track.split():
            vals = event.split(";")
            tick = int(vals[0])
            t[index][-1] = tick
            if not vals[1] == '':
                on = list(map(int, vals[1].split(",")))
                for pitch in on:
                    t[index][pitch] = 1
            if not vals[2] == '':
                off = list(map(int, vals[2].split(",")))
                for pitch in off:
                    t[index][pitch + pitch_num] = 1
            index += 1
        inputs.append(t[:-1])
        outputs.append(t[-1])
    return inputs, outputs

with open("data/tracks.txt") as f:
# with open("/data/tracks.txt") as f:
    data = f.readlines()
print("Processing Data")
train_x, train_y = to_roll_matrix(data)
print(train_y[0])

print("Training")
with tf.Session() as sess:
    sess.run(init)
    for j in range(epochs):
        batch_num = int(j % (len(train_x) / batch_size))
        train_dict = {x: train_x[batch_num * batch_size:(batch_num + 1) * batch_size],
                      y: train_y[batch_num * batch_size:(batch_num + 1) * batch_size]}
        sess.run(train, feed_dict=train_dict)
        loss = sess.run(loss, feed_dict=train_dict)
        print(loss)
        output = sess.run(tf.nn.softmax(out, dim=2), feed_dict=train_dict)
        print(output[0])