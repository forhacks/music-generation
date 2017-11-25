import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np

hidden_size = 50
out_size = 128
epochs = 1000
num_layers = 3


def process_data(text):
    chars = list(text)
    arr = [[ord(char)] for char in chars if ord(char) < out_size]
    f_x = arr[:-1]
    f_y = np.zeros((len(f_x), out_size))
    for i in range(len(f_y)):
        f_y[i][arr[i + 1]] = 1
    return [f_x, f_y]
# with floydhub
# with open("rnn/text.txt") as f:
with open("text.txt") as f:
    data = " ".join(f.readlines())

train_x, train_y = process_data(data)

seq_len = len(train_x)


print("Making graph")
x = tf.placeholder("float", [None, seq_len, 1])
y = tf.placeholder("float", [None, seq_len, out_size])
x = tf.nn.l2_normalize(x, 0)

layers = []
for l in range(num_layers):
    layers.append(rnn.BasicLSTMCell(hidden_size))
cell = rnn.MultiRNNCell(layers)

x = tf.unstack(x, seq_len, 1)
output, current_state = rnn.static_rnn(cell, x, dtype=tf.float32)
outputs = tf.transpose(output, [1, 0, 2])

w = tf.Variable(tf.truncated_normal([hidden_size, out_size], stddev=0.01))
b = tf.Variable(tf.truncated_normal([out_size], stddev=0.01))

out = [tf.matmul(output, w) + b for output in outputs]
loss = tf.reduce_sum(tf.losses.softmax_cross_entropy(y, out))

train = tf.train.AdagradOptimizer(0.3).minimize(loss)
init = tf.global_variables_initializer()


def to_str(outputs):
    string = ""
    for probs in outputs:
        string += chr(np.argmax(probs))
    return string

with tf.Session() as sess:
    sess.run(init)
    train_dict = {x: train_x, y: train_y}
    for j in range(epochs):
        sess.run(train, feed_dict=train_dict)
        print(sess.run(loss, feed_dict=train_dict))
        output = sess.run(tf.nn.softmax(out), feed_dict=train_dict)
        print(to_str(output))
