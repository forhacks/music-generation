from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
import numpy as np

max_batch_len = 50
batch_len = max_batch_len - 1
num_features = 13
epochs = 300
num_hidden = 128
dropout = 0.2
num_layers = 2
batch_size = 100


def make_model(x, y, num_features, batch_len, epochs):
    model = Sequential()
    model.add(LSTM(num_hidden, return_sequences=True, input_shape=(batch_len, num_features)))
    model.add(Dropout(dropout))
    for _ in range(num_hidden - 1):
        model.add(LSTM(128, return_sequences=True))
        model.add(Dropout(dropout))
    model.add(Dense(num_features))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adagrad')

    model.fit(x, y, batch_size=batch_size, epochs=epochs)

    return model


def is_numeric(string):
    try:
        _ = int(string)
    except ValueError:
        return False
    return True


def to_onehot(arr):
    result = []
    for token in arr:
        if is_numeric(token):
            res = [0] * 13
            res[int(token)] = 1
            result.append(res)
        elif token == ';':
            result.append([0] * 10 + [1, 0, 0])
        elif token == ',':
            result.append([0] * 10 + [0, 1, 0])
        elif token == ' ':
            result.append([0] * 10 + [0, 0, 1])
    return result


def process_data(data):
    tracks = data.split('\n')
    arr = []
    for track in tracks:
        if len(track) > max_batch_len:
            continue
        inputs = to_onehot(track)
        arr.append(inputs)
    arr = np.array([[[0] * 13] * (max_batch_len - len(t_x)) + t_x for t_x in arr])
    return np.array(arr[::, :-1:, ::], dtype=np.float), np.array(arr[::, 1::, ::], dtype=np.float)


def to_str(out_arr):
    values = np.argmax(out_arr, axis=2)
    batches = []
    for batch in values:
        string = ""
        for num in batch:
            if num < 10:
                string += str(num)
            if num == 10:
                string += ";"
            if num == 11:
                string += ","
            if num == 12:
                string += " "
        batches.append(string)
    return batches


# with open("test.txt") as f:
with open("/data/tracks.txt") as f:
    data = f.read()
print("Processing Data")
train_x, train_y = process_data(data)
print("Training")
model = make_model(train_x, train_y, num_features, batch_len, epochs)
model.predict([[0] * 13])
