import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
import math

event_len = 40
pitch_num = 120
tick_num = 12
data_size = pitch_num * 2 + tick_num
hidden_size = 300
epochs = 20000
num_layers = 2
batch_size = 16

print("Making Graph")
x = tf.placeholder("float", [None, event_len, data_size])

layers = []
for l in range(num_layers):
    layers.append(rnn.BasicLSTMCell(hidden_size))
cell = rnn.MultiRNNCell(layers)

print("Feeding Inputs Into LSTM Layers")
timesteps = tf.unstack(x, event_len, 1)
lstm_outputs, states = rnn.static_rnn(cell, timesteps, dtype=tf.float32)

# w = tf.Variable(tf.truncated_normal([hidden_size, data_size], stddev=0.1))
# b = tf.Variable(tf.truncated_normal([data_size], stddev=0.1))
w = tf.Variable(tf.zeros([hidden_size, data_size]))
b = tf.Variable(tf.zeros([data_size]))

print("Feeding Values Into Final Layer")
out = tf.nn.sigmoid(tf.matmul(lstm_outputs[-1], w) + b)

init = tf.global_variables_initializer()

saver = tf.train.Saver()


def process_output(data):
    string = ""
    for i in range(tick_num):
        if data[pitch_num * 2 + i] == 1:
            string += str((2 ** i) * 3) + ";"
            break
    for i in range(pitch_num):
        if data[i] == 1:
            string += str(i) + ","
    if string[-1] == ",":
        string = string[:-1]
    string += ";"
    for i in range(pitch_num):
        if data[pitch_num + i] == 1:
            string += str(i) + ","
    if string[-1] == ",":
        string = string[:-1]
    return string


def to_roll_matrix(data):
    result = []
    for track in data:
        t = []
        for event in track.split():
            arr = [0 for _ in range(data_size)]
            vals = event.split(";")
            tick = int(vals[0])
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
        result.append(t)
    return result

with tf.Session() as sess:
    # sess.run(init)
    saver.restore(sess, "trained/model.ckpt-3692")
    # data = [[0] * data_size for _ in range(event_len)]
    data = to_roll_matrix(
        ["0;77,62; 384;79;77 192;77;79 192;76;77 192;74;76 192;60,76;74,62 768;57;76 384;58,62;57,60 576;62;62 192;64;62 192;65;64 192;57,61;65,58 384;81,69; 384;76,69;81,69,61 384;50,77,69;57,76,69 384;77,62;50,77 384;60,76;77,62 384;74,58,69;60,76,69 384;67;69 192;73;74 192;74;73 192;76;74 192;73,69,57;58,67,76 768;76;73,69 192;76;76 192;77,62;57,76 768;50,77;77,62 192;74;77 192;55,71;74,50 576;53;55 192;52,79;53,71 192;50,77;52,79 192;48,76;50,77 384;77,53;48,76 576;48,76;77,53 192;76,55;48,76 384;74;76 384;43;55 384;72,48;74,43 384;60;48 384;;72 192;58,60;60 192;65,57;58,60 576;63;65 192;63;63 192;63;63 192;58,62;"]
    )[0][:event_len]
    tf.Print([data], w)
    tf.Print([data], b)
    # print(data.shape)
    data = [[0] * data_size for _ in range(event_len - len(data))] + data
    # data = np.random.randint(2, size=(event_len, data_size)).astype(float).tolist()
    while True:
        print(data)
        output = sess.run(out, feed_dict={x: [data]})[0]
        print(output)
        print(process_output(np.rint(output)))
        data = data[1:] + [output.tolist()]


