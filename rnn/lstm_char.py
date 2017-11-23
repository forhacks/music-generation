import tensorflow as tf
import numpy as np

hidden_size = 50
out_size = 128
epochs = 1000


def process_data(text):
    chars = list(text)
    arr = [[ord(char)] for char in chars if ord(char) < out_size]
    f_x = arr[:-1]
    f_y = np.zeros((len(f_x), out_size))
    for i in range(len(f_y)):
        f_y[i][arr[i + 1]] = 1
    return [f_x, f_y]

with open("rnn/text.txt") as f:
    data = " ".join(f.readlines())

train_x, train_y = process_data(data)

seq_len = len(train_x)


print("Making graph")
x = tf.placeholder("float", [seq_len, 1])
y = tf.placeholder("float", [seq_len, out_size])
x = tf.nn.l2_normalize(x, 0)

lstm = tf.contrib.rnn.core_rnn_cell.BasicLSTMCell(hidden_size)

hidden_state = tf.zeros([seq_len, hidden_size])
current_state = tf.zeros([seq_len, hidden_size])
state = hidden_state, current_state

w = tf.Variable(tf.truncated_normal([hidden_size, out_size], stddev=0.01))
b = tf.Variable(tf.truncated_normal([out_size], stddev=0.01))

output, state = lstm(x, state)
out = tf.matmul(output, w) + b
loss = tf.reduce_sum(tf.losses.softmax_cross_entropy(y, out))

train = tf.train.AdamOptimizer(0.01).minimize(loss)
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
