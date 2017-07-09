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
#from model import DKTnet
from DKT import DKTnet
from keras.preprocessing import sequence
import pdb
import pickle
from dataAssist import DataAssistMatrix, student

#fn = 'data.pkl'
#with open(fn, 'rb') as f:
#    data = pickle.load(f)
#    print("Load students\' data succeeded!")
data = DataAssistMatrix()
data.build()
batch_size = 64
input_dim_order =  int(data.max_questionID + 1) #consider whether we need plus 1
input_dim = 2 * input_dim_order
epoch = 100
hidden_layer_size = 512
# x_train =
# y_train =
# y_train_order =

# x_test =
# y_test =
# y_test_order =

'''Training part starts from now'''
x_train = []
y_train = []
y_train_order = []
num_student = 0
for student in data.trainData:
    num_student += 1
    if num_student % 200 ==0:
        print (num_student,' ',num_student/46051.)
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
#x_train = sequence.pad_sequences(x_train, maxlen=1000, dtype='float64',padding='post', truncating='post', value=-1.)
print ('preprocessing finished')
x_train = np.array(x_train)
y_train = np.array(y_train)
y_train_order = np.array(y_train_order)
#now the dimensions are samples, time_length, questions
#this procedure is for matching the pred answer of next question to the groundtruth of next question.
pdb.set_trace()
x_train = x_train[:,:-1,:]
y_train = y_train[:,1:,:]
y_train_order = y_train_order[:,1:,:]
#'''Testing part starts from now'''
#x_test = []
#y_test = []
#y_test_order = []
#
#num_student = 0
#for student in data.testData:
#    num_student += 1
#    x_single_test = np.zeros([input_dim, data.longest])
#    y_single_test = np.zeros([1, data.longest])
#    y_single_test_order = np.zeros([input_dim_order, data.longest])
#
#    for i in  range(student.n_answers):
#        if student.correct[i] == 1.: # if correct
#            x_single_test[student.ID[i]*2-1, i] = 1.
#        elif student.correct[i] == 0.: # if wrong
#            x_single_test[student.ID[i]*2, i] = 1.
#        else:
#            print (student.correct[i])
#            print ("wrong length with student's n_answers or correct")
#        y_single_test[0, i] = student.correct[i]
#        y_single_test_order[student.ID[i], i] = 1.
#    for i in  range(data.longest-student.n_answers):
#        x_single_test[:,student.n_answers + i] = -1
#        y_single_test[:,student.n_answers + i] = -1
#        #notice that the padding value of order is still zero.
#        y_single_test_order[:,student.n_answers + i] = 0
#    x_single_test = np.transpose(x_single_test)
#    y_single_test = np.transpose(y_single_test)
#    y_single_test_order = np.transpose(y_single_test_order)
#    x_test.append(x_single_test)
#    y_test.append(y_single_test)
#    y_test_order.append(y_single_test_order)
#print ("test num students", num_student)
##x_train = sequence.pad_sequences(x_train, maxlen=1000, dtype='float64',padding='post', truncating='post', value=-1.)
#print ('preprocessing finished')
#x_test = np.array(x_test)
#y_test = np.array(y_test)
#y_test_order = np.array(y_test_order)
##this procedure is for matching the pred answer of next question to the groundtruth of next question.
#x_test = x_test[:,:-1,:]
#y_test = y_test[:,1:,:]
#y_test_order = y_test_order[:,1:,:]
#pdb.set_trace()
#model = DKTnet(input_dim, input_dim_order,batch_size, epoch,
        #x_train, y_train, y_train_order,
        #x_test, y_test, y_test_order)
model = DKTnet(input_dim, input_dim_order, hidden_layer_size,
        batch_size, epoch, np.array(x_train), np.array(y_train), np.array(y_train_order))

model.build()
