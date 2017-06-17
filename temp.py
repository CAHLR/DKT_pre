from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from midi_to_statematrix import noteStateMatrixToMidi
from theano import tensor as T
import numpy as np
import copy
import cPickle
import datetime
from allchords import get_all_chords
from util import sample_by_pr, sample_bin

mode = [1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1]

numbers = 4 + 6 * 108 + 1 + 12 + 13
maxlen = 8
nSample = 1000

diversity = 3


def predict(model, start):
    X = start
    # ret = start
    ret = []
    chord_list = []
    last_ret = []
    for i in range(128):
        y = model.predict(np.array([X]), batch_size=16)
        last_ret = sample(y, X[-1], last_ret, chord_list)[0]
        ret.append(last_ret)
        print y
        X = np.concatenate((X[1:], y))
        # print y
    return ret, chord_list


def compress(notes):
    notes = np.array(notes)
    sz = notes.shape
    ret = np.reshape(notes, (sz[0], sz[1] / 2, 2))
    return ret


def my_loss(y_true, y_pred):
    return T.nnet.categorical_crossentropy(y_pred[:, :4], y_true[:, :4]) + \
    T.nnet.categorical_crossentropy(y_pred[:, 4:112], y_true[:, 4:112]) + \
    T.nnet.categorical_crossentropy(y_pred[:, 112:220], y_true[:, 112:220])+ \
    T.nnet.categorical_crossentropy(y_pred[:, 220:328], y_true[:, 220:328])+ \
    T.nnet.categorical_crossentropy(y_pred[:, 328:436], y_true[:, 328:436])+ \
    T.nnet.categorical_crossentropy(y_pred[:, 436:544], y_true[:, 436:544])+ \
    T.nnet.categorical_crossentropy(y_pred[:, 544:652], y_true[:, 544:652])+ \
    T.nnet.categorical_crossentropy(y_pred[:, 652:653], y_true[:, 652:653])+ \
    T.nnet.categorical_crossentropy(y_pred[:, 653:665], y_true[:, 653:665])+ \
    T.nnet.categorical_crossentropy(y_pred[:, 665:], y_true[:, 665:])

def build(X_train, Y_train):
    # order = [0, 3, 4, 3, 5, 3, 4, 3]
    order = [0]
    for i in range(5):
        t = int(np.random.random() * 6 - 1e-6)
        order.append(t)

    model = Sequential()

    lstm = LSTM(numbers*2 , return_sequences=False, input_shape=(maxlen, numbers))
    model.add(lstm)
    model.add(Dropout(0.5))
    model.add(Activation('sigmoid'))
    # model.add(Dense(numbers))

    # model.add(LSTM(numbers , return_sequences=False))
    # model.add(Dropout(0.5))
    model.add(Dense(numbers))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    model.compile(loss=my_loss, optimizer='rmsprop')

    X_train = np.array(X_train)       
    Y_train = np.array(Y_train)

    print np.array(X_train).shape, np.array(Y_train).shape
    for i in range(100):
        batch_size = 100
        print i, "\t",
        model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=10, verbose=1)

        model.save_weights('../result/model/model' + str(i) + '.hdf5')
        # model.load_weights('../result/model/model' + str(i) + '.hdf5')
        evaluate = model.evaluate(X_train, Y_train, batch_size=batch_size)
        print evaluate




if __name__ == "__main___":
    order = [0, 5, 1, 3, 4, 2, 4]
    ret = generate_chord([0,7,12+9, 12+4], order) #C G Am Em
    noteStateMatrixToMidi(compress(ret), "../result/logs/chord" + datetime.datetime.strftime(datetime.datetime.now(), ' %H:%M:%s'))


if __name__ == "__main__":
    data = cPickle.load(open('../result/middle/rhythm10.pkl', 'rb'))

    X_train = []
    Y_train = []
    for piece in data:
        piece = piece.toarray().tolist()
        for i in range(0, len(piece) - maxlen - 1, maxlen):
            x = []
            for j in range(maxlen):
                x.append(piece[i + j])
            X_train.append(x)
            Y_train.append(piece[i + maxlen])

    cPickle.dump(X_train[0], open('../result/model/start.pkl','wb'))
    print np.array(X_train).shape, np.array(Y_train).shape
    build(X_train, Y_train)








