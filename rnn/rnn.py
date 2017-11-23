import tensorflow as tf
from gensim.models import Word2Vec
import numpy as np
import re

print("Loading Word2Vec")
w2v = Word2Vec.load("/Users/frank/PycharmProjects/study-tool-back/trained/w2v/trained.w2v")

epochs = 100
seq_len = 10
size = 300
hidden_size1 = 100
hidden_size2 = 50
stddev = 0.001
batch_size = 20
out_size = len(w2v.wv.vocab)
articles = ["a", "the", "an", "i"]

print("Loading Graph")

x = tf.placeholder("float", [seq_len, None, size])
y = tf.placeholder("float", [None, size])

w1 = tf.Variable(tf.truncated_normal([size, hidden_size1], stddev=stddev))
b1 = tf.Variable(tf.truncated_normal([hidden_size1], stddev=stddev))

w2 = tf.Variable(tf.truncated_normal([hidden_size1, hidden_size2], stddev=stddev))
b2 = tf.Variable(tf.truncated_normal([hidden_size2], stddev=stddev))

w3 = tf.Variable(tf.truncated_normal([hidden_size2, size], stddev=stddev))
b3 = tf.Variable(tf.truncated_normal([size], stddev=stddev))

h1 = tf.Variable(tf.truncated_normal([hidden_size1, hidden_size1], stddev=stddev))
h2 = tf.Variable(tf.truncated_normal([hidden_size2, hidden_size2], stddev=stddev))

hidden1 = None
hidden2 = None

for i in range(seq_len):
    window = x[i]
    if hidden1 is None:
        hidden1 = tf.nn.tanh(tf.matmul(window, w1)) + b1
        hidden2 = tf.nn.tanh(tf.matmul(hidden1, w2)) + b2
    else:
        hidden1 = tf.nn.tanh(tf.matmul(window, w1) + tf.matmul(hidden1, h1) + b1)
        hidden2 = tf.nn.tanh(tf.matmul(hidden1, w2) + tf.matmul(hidden2, h2) + b2)

out = tf.nn.sigmoid(tf.matmul(hidden2, w3) + b3)
loss = tf.reduce_sum(tf.losses.cosine_distance(tf.nn.l2_normalize(y, 0), tf.nn.l2_normalize(out, 0), dim=1))

train = tf.train.AdamOptimizer(0.01).minimize(loss)

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
        data = " ".join(f.readlines())

    test = "We know each other so well that sometimes when we talk we finish each others [blank]"
    train_x, train_y = process_data(data)
    print("Training")
    for i in range(epochs):
        sess.run(train, feed_dict={x: train_x, y: train_y})
        print(sess.run(loss, feed_dict={x: train_x, y: train_y}))
    print("Generating")
    for i in range(100):
        test_x, _ = process_data(test)
        val = sess.run(out, feed_dict={x: test_x[::, i:i+1:, ::]})
        word = w2v.most_similar(positive=val, topn=1)[0][0]
        test = test.rsplit(' ', 1)[0] + " " + word + " [blank]"
        print(test)

