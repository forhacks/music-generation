import tensorflow as tf
from gensim.models import Word2Vec
import numpy as np
import re
epochs = 200
seq_len = 15
size = 300
hidden_size = 400
stddev = 0.001
articles = ["a", "the", "an"]

print("Loading Graph")

x = tf.placeholder("float", [seq_len, None, size])
y = tf.placeholder("float", [None, size])

w1 = tf.Variable(tf.truncated_normal([size, hidden_size], stddev=stddev))
b1 = tf.Variable(tf.truncated_normal([hidden_size], stddev=stddev))

w2 = tf.Variable(tf.truncated_normal([hidden_size, size], stddev=stddev))
b2 = tf.Variable(tf.truncated_normal([size], stddev=stddev))

h1 = tf.Variable(tf.truncated_normal([hidden_size, hidden_size], stddev=stddev))

hidden = None

for i in range(seq_len):
    window = x[i]
    if hidden is None:
        hidden = tf.nn.tanh(tf.matmul(window, w1))
    else:
        hidden = tf.nn.tanh(tf.matmul(window, w1) + tf.matmul(hidden, h1) + b1)

out = tf.nn.sigmoid(tf.matmul(hidden, w2) + b2)
loss = tf.reduce_sum(tf.losses.cosine_distance(tf.nn.l2_normalize(y, 0), tf.nn.l2_normalize(out, 0), dim=1))

train = tf.train.AdamOptimizer(0.001).minimize(loss)

init = tf.global_variables_initializer()


def process_data(dat):
    dat = dat.lower().split()
    dat = [re.sub("[^A-Za-z]", "", word) for word in dat]
    dat = [w2v[word] for word in dat if word not in articles]
    t_x = [dat[j:j + seq_len] for j in range(len(dat) - seq_len)]
    t_x = np.array(t_x).swapaxes(0, 1)
    t_y = np.array(dat[seq_len:])
    return [t_x, t_y]

with tf.Session() as sess:
    sess.run(init)

    with open("data/text.txt") as f:
        data = f.readlines()

    test = """One method of solving climate change that is better than [blank]"""
    print("Loading Word2Vec")
    w2v = Word2Vec.load("/Users/frank/PycharmProjects/study-tool-back/trained/w2v/trained.w2v")
    train_x, train_y = process_data(data)
    print("Training")
    for i in range(epochs):
        sess.run(train, feed_dict={x: train_x, y: train_y})
        print(sess.run(loss, feed_dict={x: train_x, y: train_y}))
    for i in range(100):
        test_x, _ = process_data(test)
        val = sess.run(out, feed_dict={x: test_x[::, i:i+1:, ::]})
        word = w2v.most_similar(positive=val, topn=1)[0][0]
        test = test.rsplit(' ', 1)[0] + " " + word + " [blank]"
        print(test)

