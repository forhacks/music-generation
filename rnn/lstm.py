import tensorflow as tf
from gensim.models import Word2Vec
import numpy as np
import re

print("Loading Word2Vec")
w2v = Word2Vec.load("/Users/frank/PycharmProjects/study-tool-back/trained/w2v/trained.w2v")

with open("data/text.txt") as f:
    data = " ".join(f.readlines())
words = re.sub("[^A-Za-z]", "", data).split()


def word2dictionary(dat):
    # iterate thru all words and assign id
    dict = {}
    count = 0
    for word in dat:
        if word not in dict:
            dict[word] = count
            count += 1
    return dict
lexicon = word2dictionary(words)


epochs = 100
seq_len = len(data) - 1
size = 300
hidden_size1 = 100
hidden_size2 = 50
stddev = 0.001
out_size = len(lexicon)
articles = ["a", "the", "an", "i"]


def process_data(dat):
    dat = dat.lower().split()
    dat = [re.sub("[^A-Za-z]", "", word) for word in dat]
    dat = [w2v[word] for word in dat if word not in articles]
    t_x = [dat[j:j + seq_len] for j in range(len(dat) - seq_len)]
    t_x = np.array(t_x).swapaxes(0, 1)
    t_y = np.array(dat[seq_len:])
    return [t_x, t_y]

train_x = process_data(data)[:-1]
train_y = np.zeros((len(data) - 1, len(lexicon)))
train_y[lexicon[data[1:]]] = 1


x = tf.placeholder([seq_len, size])
y = tf.placeholder([seq_len, out_size])

lstm = tf.contrib.rnn.BasicLSTMCell(hidden_size1)
hidden_state = tf.zeros([seq_len, lstm.state_size])
current_state = tf.zeros([seq_len, lstm.state_size])

w = tf.truncated_normal([hidden_size1, out_size])

state = hidden_state, current_state

loss = 0

for word in x:
    output, state = lstm(word, state)
    out_layer = tf.matmul(output, w)
    loss += tf.losses.sigmoid_cross_entropy(y, out_layer)

train = tf.train.AdamOptimizer(tf.reduce_sum(loss))

init = tf.global_variables_initializer()

with tf.Session as sess:
    sess.run(init)
    for i in epochs:
        sess.run(train, feed_dict={x: train_x, y: train_y})
