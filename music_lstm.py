import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
import math
from collections import deque
import time

event_len = 40
pitch_num = 120
tick_num = 12
data_size = pitch_num * 2 + tick_num
hidden_size = 300
epochs = 20000
num_layers = 2
batch_size = 16
start = time.time()

print("Making Graph")
x = tf.placeholder("float", [None, event_len, data_size])
y = tf.placeholder("float", [None, data_size])

layers = []
for l in range(num_layers):
    layers.append(rnn.BasicLSTMCell(hidden_size))
cell = rnn.MultiRNNCell(layers)

print("Feeding Inputs Into LSTM Layers")
timesteps = tf.unstack(x, event_len, 1)
lstm_outputs, states = rnn.static_rnn(cell, timesteps, dtype=tf.float32)

w = tf.Variable(tf.truncated_normal([hidden_size, data_size], stddev=0.1))
b = tf.Variable(tf.truncated_normal([data_size], stddev=0.1))

print("Feeding Values Into Final Layer")
out = tf.matmul(lstm_outputs[-1], w) + b
loss = tf.reduce_sum(tf.losses.sigmoid_cross_entropy(y, out))

print('Computing Gradients')
train = tf.train.AdamOptimizer(0.0001).minimize(loss)
init = tf.global_variables_initializer()

saver = tf.train.Saver()


def is_power_2(num):
    num = int(num)
    return (num & (num - 1)) == 0


def to_roll_matrix(data):
    inputs = []
    outputs = []
    for track in data:
        fail = False
        t = []
        for event in track.split():
            arr = [0 for _ in range(data_size)]
            vals = event.split(";")
            tick = int(vals[0])
            if tick % 3 != 0 or not is_power_2(tick / 3):
                fail = True
                break
            tick = int(math.log((tick / 3), 2)) if tick != 0 else 0
            arr[-(tick_num - tick)] = 1
            if not vals[1] == '':
                on = list(map(int, vals[1].split(",")))
                for pitch in on:
                    arr[pitch] = 1
            if not vals[2] == '':
                off = list(map(int, vals[2].split(",")))
                for pitch in off:
                    arr[pitch + pitch_num] = 1
            t.append(arr)
        if not fail:
            values = list(window(t))
            inputs.extend(values[:-1])
            outputs.extend(t[event_len:])
    return inputs, outputs


def window(seq, n=event_len, samples=999999999999):
    it = iter(seq)
    win = deque((next(it, None) for _ in range(n)), maxlen=n)
    yield list(win)
    append = win.append
    i = 0
    for e in it:
        if i == samples:
            break
        i += 1
        append(e)
        yield list(win)


# with open("data/tracks.txt") as f:
with open("/data/tracks.txt") as f:
    data = f.readlines()
print("Processing Data")
train_x, train_y = to_roll_matrix(data)
# print(train_y[0])
# print(train_x[1][-1])
print(len(train_y))

print("Training")
minimum = 99999
with tf.Session() as sess:
    sess.run(init)
    for j in range(epochs):
        batch_num = int(j % (len(train_x) / batch_size))
        train_dict = {x: train_x[batch_num * batch_size:(batch_num + 1) * batch_size],
                      y: train_y[batch_num * batch_size:(batch_num + 1) * batch_size]}
        sess.run(train, feed_dict=train_dict)
        curr_loss = sess.run(loss, feed_dict=train_dict)
        print("Loss: " + str(curr_loss))
        print("Epoch: " + str(j))
        print(sess.run(tf.nn.sigmoid()))
        if curr_loss < minimum and time.time() - start > 120:
            minimum = curr_loss
            # save_path = saver.save(sess, "trained/model.ckpt", global_step=j)
            save_path = saver.save(sess, "/output/model.ckpt", global_step=j)
