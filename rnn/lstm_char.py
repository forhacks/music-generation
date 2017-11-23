import tensorflow as tf
import numpy as np


def process_data(text):
    chars = list(text)
    arr = [ord(char) for char in chars]
    return [arr[:-1], arr[1:]]

with open("../data/text.txt") as f:
    data = " ".join(f.readlines())

train_x, train_y = process_data(data)

seq_len = len(train_x)
hidden_size = 50
out_size = 128
epochs = 1000

x = tf.placeholder("float", [seq_len, 1])
y = tf.placeholder(tf.int32, [seq_len, out_size])

lstm = tf.contrib.rnn.core_rnn_cell.LSTMCell(hidden_size)

hidden_state = tf.zeros([seq_len, hidden_size])
current_state = tf.zeros([seq_len, hidden_size])
state = hidden_state, current_state

w = tf.Variable(tf.truncated_normal([hidden_size, out_size], stddev=0.001))
b = tf.Variable(tf.truncated_normal([out_size], stddev=0.001))

output, state = lstm(x, state)
out = tf.matmul(output, w) + b
loss = tf.reduce_sum(tf.losses.softmax_cross_entropy(out, y))

train = tf.train.AdamOptimizer(0.01).minimize(loss)
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for j in range(epochs):
        sess.run(train, feed_dict={x: train_x, y: train_y})
