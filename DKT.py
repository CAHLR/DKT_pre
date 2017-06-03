from keras.models import Model
from keras.layers import Input, Dropout, Masking, Dense, Embedding
from keras.layers import Embedding
from keras.layers.core import Flatten, Reshape
from keras.layers import LSTM
from keras.layers.recurrent import SimpleRNN
from keras.layers import merge
from keras.layers.merge import multiply
from keras.callbacks import EarlyStopping
from keras import backend as K
from theano import tensor as T
from theano import config
from theano import printing
from theano import function
from keras.layers import Lambda
import theano
import numpy as np
import pdb
from math import sqrt
from keras.callbacks import Callback

"""
To calculate and print the average rmse on the validation set after every epoch
"""
class TestCallback(Callback):
    
    def __init__(self, test_data):
        self.x_test, self.y_test_order, self.y_test = test_data

    def on_epoch_end(self, epoch, logs={}):
        
        y_pred = self.model.predict([self.x_test, self.y_test_order])
        avg_rmse = self.rmse_masking(self.y_test, y_pred)
        print('\nTesting avg_rmse: {}\n'.format(avg_rmse))
        
    def rmse_masking(self, y_true, y_pred):
        
        mask_matrix = np.sum(self.y_test_order, axis=2).flatten()
        num_users, max_responses = np.shape(self.x_test)[0], np.shape(self.x_test)[1]
        y_pred = y_pred.flatten()
        y_true = y_true.flatten()
        rmse = [] 
        for user in range(num_users): 
            diff_sq, response = 0, 0
            for i in range(user * max_responses, (user + 1) * max_responses):
                if mask_matrix[i] == 0:
                    break
                diff_sq += (y_true[i] - y_pred[i]) ** 2 
                response += 1
            if response != 0:
                rmse.append(sqrt(diff_sq/float(response)))   
        return sum(rmse)/float(len(rmse))    

class DKTnet():
    
    def __init__(self, input_dim, input_dim_order, hidden_layer_size, batch_size, epoch,
        x_train, y_train, y_train_order,
        x_test=[], y_test=[], y_test_order=[]):
        
        ## input dim is the dimension of the input at one timestamp (dimension of x_t) 
        self.input_dim = int(input_dim)
        ## input_dim_order is the dimension of the one hot representation of problem to
        ## check the order of occurence of responses according to timestamp
        self.input_dim_order = int(input_dim_order)
        
        self.hidden_layer_size = hidden_layer_size
        self.batch_size = int(batch_size)
        self.epoch = int(epoch)
        
        ## xtrain is a 3D matrix of size ( samples * number of timestamp * dimension of input vec (x_t) )
        ## in cognitive tutor # of students * # total responses * # input_dim
        self.x_train = x_train
        ## y_train is a matrix of ( samples * one hot representation according to problem output value at each timestamp (y_t) )
        self.y_train = y_train
        ## y_train_order is the one hot representation of problem according to timestamp starting from 
        ## t=1 if training starts at t=0
        self.y_train_order = y_train_order
        
        self.x_test = x_test
        self.y_test = y_test
        self.y_test_order = y_test_order
        
        self.users = np.shape(x_train)[0]
        self.validation_split = 0.2
        print ("Initialization Done")
    
    def build(self):
        
        ## first layer for the input (x_t)
        x = Input(batch_shape = (None, None, self.input_dim), name='x')
        # masked = (Masking(mask_value= -1, input_shape = (None, None, self.input_dim)))(x)
        lstm_out = SimpleRNN(self.hidden_layer_size, input_shape = (None, None, self.input_dim), return_sequences = True)(x)
        dense_out = Dense(self.input_dim_order, input_shape = (None, None, self.hidden_layer_size), activation='sigmoid')(lstm_out)
        y_order = Input(batch_shape = (None, None, self.input_dim_order), name = 'y_order')
        merged = multiply([dense_out, y_order])

        def reduce_dim(x):
            x = K.max(x, axis = 2, keepdims = True)
            return x
        
        def reduce_dim_shape(input_shape):
            shape = list(input_shape)
            shape[-1] = 1
            print ("reduced_shape", shape)
            return tuple(shape)
        
        earlyStopping = EarlyStopping(monitor='val_loss', patience=2, verbose=0, mode='auto')
        reduced = Lambda(reduce_dim, output_shape = reduce_dim_shape)(merged)
        model = Model(inputs=[x,y_order], outputs=reduced)
        model.compile( optimizer = 'rmsprop',
                        loss = 'binary_crossentropy',
                        metrics=['accuracy'])

                        #binary_crossentropy
                        #1 prob
                        #next question
                        #callbacks =[ histories])
                        #callbacks=[histories, model_check, lr])
                        #rmc->auc
                        #1value y_train

        #histories = my_callbacks.Histories()
        #print "x_test",np.shape(self.x_test)
        #print "y_test_order",np.shape(self.y_test_order)

        #layer_name = 'my_layer'
        #intermediate_layer_model = Model(inputs = [x,y_order],outputs = reduced)
        #intermediate_output = intermediate_layer_model.predict([self.x_train,self.y_train_order])
        #print 'reduced',intermediate_output.shape
        #pdb.set_trace()
        
        model.fit([self.x_train, self.y_train_order], self.y_train, batch_size = self.batch_size, \
                  epochs=self.epoch, \
                  callbacks = [ earlyStopping, \
                                TestCallback((self.x_train[int((1-self.validation_split)*self.users):], \
                                self.y_train_order[int((1-self.validation_split)*self.users):], \
                                self.y_train[int((1-self.validation_split)*self.users):]))  ], \
                  validation_split = self.validation_split, shuffle = True) 
        
        #for layer in model.layers:
        #    weights = layer.get_weights()
        #    print (weights)
        #for layer in model.layers:
        #        print (np.shape(layer.get_weights()))            
        #validation_data=([self.x_train,self.y_train_order],self.y_train))
        #score = model.evaluate([self.x_test, self.y_test_order], self.y_test, batch_size= self.batch_size)
        #print (score)
        # print (model.predict([self.x_train, self.y_train_order]))
        
        
