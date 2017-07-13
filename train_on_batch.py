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
from DKT import *
from keras.preprocessing import sequence
import pdb
import my_callbacks
import pickle
from dataAssist import DataAssistMatrix, student
import random
import sys

data = DataAssistMatrix()
data.build()
batch_size = 2
input_dim_order =  int(data.max_questionID + 1)
input_dim = 2 * input_dim_order
epoch = 10
hidden_layer_size = 512
validation_slpit = 0.2 #extract validation data from training data

validation_data = [] # sample validation data based on validation_split on every epoch
train_data = []

for student in data.trainData:
    if random.uniform(0,1)<validation_slpit:
        validation_data.append(student)
    else: train_data.append(student)

print('The total size of raw data is: ', sys.getsizeof(data.trainData))
data.trainData = [] # To save memory.

DKTmodel = DKTnet(input_dim, input_dim_order, hidden_layer_size, batch_size, epoch)
DKTmodel.build_train_on_batch()

sum_acc = [] # using for earlystopping
sum_rmse = []# using for earlystopping

for epo in range(epoch):
    '''Initializing'''
    x_train = []
    y_train = []
    y_train_order = []
    num_student = 0 # num of TRAINING student in each epoch

    print ('Now starts the ',epo+1,'th epoch')

    '''Training part starts from now'''
    random.shuffle(train_data)
    print('Training data is shuffled')
    for student in train_data:
        num_student += 1
        # print (num_student)
        if num_student % batch_size == 0:
            if num_student % (batch_size*10) == 0:
                print ("Training when num student is",num_student)
            x_train = np.array(x_train)
            y_train = np.array(y_train)
            y_train_order = np.array(y_train_order)

            x_train = x_train[:,:-1,:]
            y_train = y_train[:,1:,:]
            y_train_order = y_train_order[:,1:,:]
            DKTmodel.train_on_batch(x_train, y_train, y_train_order)


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
    print ("validation num students", len(validation_data))



    '''Validation part starts from now'''


    x_val = []
    y_val = []
    y_val_order = []
    num_val = 0
    y_pred_total = []
    y_true_total = []
    rmse = []
    acc = []
    callback = TestCallback()

    for student in validation_data:
        num_val += 1
        if num_val % batch_size == 0:
            if num_val % (batch_size*10) == 0:
                print ("Predicting when num student is",num_val)
            x_val = np.array(x_val)
            y_val = np.array(y_val)
            y_val_order = np.array(y_val_order)

            x_val = x_val[:,:-1,:]
            y_val = y_val[:,1:,:]
            y_val_order = y_val_order[:,1:,:]
            # DKTmodel = DKTnet(input_dim, input_dim_order, hidden_layer_size, batch_size, epoch,
            #     x_val, y_val, y_val_order)
            # DKTmodel.train_on_batch()

            y_pred = DKTmodel.predict(x_val,y_val_order)
            y_pred.flatten()
            y_val.flatten()
            # y_val is y_true
            tmp_rmse, tmp_acc = callback.rmse_masking_on_batch(y_val, y_pred, y_val_order)
            rmse += (tmp_rmse)
            acc += (tmp_acc)
            # y_pred_total = y_pred_total + list(y_pred)
            # y_true_total = y_true_total + list(y_val)


            x_val = []
            y_val = []
            y_val_order = []

        x_single_val = np.zeros([input_dim, data.longest])
        y_single_val = np.zeros([1, data.longest])
        y_single_val_order = np.zeros([input_dim_order, data.longest])

        for i in range(student.n_answers):
            if student.correct[i] == 1.: # if correct
                x_single_val[student.ID[i]*2-1, i] = 1.
            elif student.correct[i] == 0.: # if wrong
                x_single_val[student.ID[i]*2, i] = 1.
            else:
                print (student.correct[i])
                print ("wrong length with student's n_answers or correct")
            y_single_val[0, i] = student.correct[i]
            y_single_val_order[student.ID[i], i] = 1.

        for i in  range(data.longest-student.n_answers):
            x_single_val[:,student.n_answers + i] = -1
            y_single_val[:,student.n_answers + i] = -1
            #notice that the padding value of order is still zero.
            y_single_val_order[:,student.n_answers + i] = 0
        x_single_val = np.transpose(x_single_val)
        y_single_val = np.transpose(y_single_val)
        y_single_val_order = np.transpose(y_single_val_order)
        x_val.append(x_single_val)
        y_val.append(y_single_val)
        y_val_order.append(y_single_val_order)
    
    avg_rmse, avg_acc = sum(rmse)/float(len(rmse)), sum(acc)/float(len(acc))
    print('\nTesting avg_rmse: {}\n'.format(avg_rmse))
    print('\nTesting avg_acc: {}\n'.format(avg_acc))
    sum_acc.append(avg_acc)
    sum_rmse.append(avg_rmse)
    if len(sum_acc)>=3 and sum_acc[-1]<sum_acc[-2] and sum_acc[-2]<sum_acc[-3]: # patience is 2
        print ('sum_acc:',sum_acc)
        print ('sum_rmse:', sum_rmse)
        pdb.set_trace()

