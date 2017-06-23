# coding: utf-8
import numpy as np
import csv
import utils
from keras.models import Model
from keras.layers import Input, Dense, Dropout
from keras.layers import Embedding
from keras.layers import LSTM
from keras.layers import Merge
from theano import tensor as T
from dataAssist import DataAssistMatrix
from DKT import DKTnet
from keras.preprocessing import sequence
import pdb
import pickle
from dataAssist import DataAssistMatrix, student
fn = 'data.pkl'
with open(fn, 'rb') as f:
    data = pickle.load(f)
    print("Load students\' data succeeded!")

batch_size = 64
input_dim_order =  1
input_dim = 1
epoch = 100
hidden_layer_size = 512


'''Training part starts from now'''
x_train = []
y_train = []
num_student = 0
for student in data.trainData:
    num_student += 1
    x_single_train = np.zeros([1, data.longest])
    y_single_train = np.zeros([1, data.longest])

    for i in  range(student.n_answers):
        if student.correct[i] == 1.: # if correct
            x_single_train[0, i] = student.questionsID[i]*2-1
        elif student.correct[i] == 0.: # if wrong
            x_single_train[0, i] = student.questionsID[i]*2
        else:
            print (student.correct[i])
            print ("wrong length with student's n_answers or correct")
        y_single_train[0, i] = student.correct[i]
    for i in  range(data.longest-student.n_answers):
        x_single_train[:,student.n_answers + i] = -1
        y_single_train[:,student.n_answers + i] = -1
        #notice that the padding value of order is still zero.
    x_single_train = np.transpose(x_single_train)
    y_single_train = np.transpose(y_single_train)
    x_train.append(x_single_train)
    y_train.append(y_single_train)
print ("train num students", num_student)
print ('preprocessing finished')
x_train = np.array(x_train)
y_train = np.array(y_train)
#now the dimensions are samples, time_length, questions
#this procedure is for matching the pred answer of next question to the groundtruth of next question.
x_train = x_train[:,:-1,:]
y_train = y_train[:,1:,:]

pdb.set_trace()
model = DKTnet(input_dim, input_dim_order, hidden_layer_size,
        batch_size, epoch, np.array(x_train), np.array(y_train))

model.build()
