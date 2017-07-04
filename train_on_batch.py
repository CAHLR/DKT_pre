# coding: utf-8
import numpy as np
import csv
import utils
from keras.models import Model
from dataAssist import DataAssistMatrix
from keras.layers import Input, Dense, Dropout, Masking
from keras.layers import Embedding
from keras.layers import LSTM
from keras.layers import merge
from keras.layers import Dot
from keras import backend as K
from theano import tensor as T
from theano import config
from theano import printing
from keras.layers import Lambda
import theano
import numpy as np
import utils
import my_callbacks
import pdb
from DKT import DKTnet
from keras.preprocessing import sequence
import pdb
import my_callbacks
import pickle
from dataAssist import DataAssistMatrix, student

# fn = 'data.pkl'
# with open(fn, 'rb') as f:
#     data = pickle.load(f)
#     print("Load students\' data succeeded!")
data = DataAssistMatrix()
data.build()
batch_size = 2
input_dim_order =  int(data.max_questionID + 1)
input_dim = 2 * input_dim_order
epoch = 100
hidden_layer_size = 512

# def batch_training(x_train, y_train, y_train_order):
#     x = Input(batch_shape=(None,None, input_dim), name='x')
#     y_order = Input(shape = (None,input_dim_order), name = 'y_order')
#     masked = (Masking(mask_value= -1,input_shape=(None,None,input_dim)))(x)
#     lstm_out = LSTM(input_dim_order, input_shape = (None, None, input_dim), return_sequences = True)(masked)
#     #merged=merge([lstm_out,y_order],mode='mul')

#     def reduce_dim(x):
#         x = K.max(x,axis = 2, keepdims = True)
#         return x
#     def reduce_dim_shape(input_shape):
#         shape = list(input_shape)
#         shape[-1] = 1
#         print ("reduced_shape",shape)
#         return tuple(shape)

#     #reduced = Lambda(reduce_dim,output_shape = reduce_dim_shape)(merged)
#     model = Model(inputs = [x,y_order],outputs = masked)
#     #model = Model(inputs=[x,y_order],outputs=reduced)
#     histories = my_callbacks.Histories()
#     model.compile( optimizer = 'rmsprop',
#                     loss = 'binary_crossentropy',
#                     metrics=['accuracy'])
#     model.train_on_batch([x_train, y_train_order], y_train)
#     pdb.set_trace()
#     # model.fit([x_train, y_train_order], y_train, batch_size = batch_size,epochs=epoch, callbacks =
#     #         [histories],
#     # validation_split = 0.2, shuffle = True)


'''Training part starts from now'''
x_train = []
y_train = []
y_train_order = []
num_student = 0
for student in data.trainData:
    print ('student.n_answers ', student.n_answers)
    num_student += 1
    # print (num_student)
    if num_student % batch_size == 0:
        print ("Training when num student is",num_student)
        x_train = np.array(x_train)
        y_train = np.array(y_train)
        y_train_order = np.array(y_train_order)

        x_train = x_train[:,:-1,:]
        y_train = y_train[:,1:,:]
        y_train_order = y_train_order[:,1:,:]
        model = DKTnet(input_dim, input_dim_order, hidden_layer_size, batch_size, epoch,
            x_train, y_train, y_train_order)
        model.train_on_batch()

        x_train = []
        y_train = []
        y_train_order = []

    x_single_train = np.zeros([input_dim, data.longest])
    y_single_train = np.zeros([1, data.longest])
    y_single_train_order = np.zeros([input_dim_order, data.longest])

    for i in range(student.n_answers):
        if student.correct[i] == 1.: # if correct
            x_single_train[student.ID[i]*2-1, i] = 1.
        elif student.correct[i] == 0.: # if wrong
            x_single_train[student.ID[i]*2, i] = 1.
        else:
            print (student.correct[i])
            print ("wrong length with student's n_answers or correct")
        y_single_train[0, i] = student.correct[i]
        y_single_train_order[student.ID[i], i] = 1.

    for i in  range(data.longest-student.n_answers):
        x_single_train[:,student.n_answers + i] = -1
        y_single_train[:,student.n_answers + i] = -1
        #notice that the padding value of order is still zero.
        y_single_train_order[:,student.n_answers + i] = 0
    x_single_train = np.transpose(x_single_train)
    y_single_train = np.transpose(y_single_train)
    y_single_train_order = np.transpose(y_single_train_order)
    x_train.append(x_single_train)
    y_train.append(y_single_train)
    y_train_order.append(y_single_train_order)
print ("train num students", num_student)
pdb.set_trace()



