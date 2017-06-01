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
from keras.layers import Lambda
import theano
import numpy as np
import pdb

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
        print ("Initialization Done")

    def build(self):
        
        ## first layer for the input (x_t)
        x = Input(batch_shape = (None, None, self.input_dim), name='x')
        
        ## masked layer to skip timestamp (t) when all the values of input vector (x_t) are -1 
        masked = (Masking(mask_value= -1, input_shape = (None, None, self.input_dim)))(x)
        lstm_out = SimpleRNN(self.hidden_layer_size, input_shape = (None, None, self.input_dim), return_sequences = True)(masked)
        lstm_out = Dense(self.input_dim_order, input_shape = (None, None, self.hidden_layer_size), activation='sigmoid')(lstm_out)
        y_order = Input(batch_shape = (None, None, self.input_dim_order), name = 'y_order')
        merged = multiply([lstm_out, y_order])

        def reduce_dim(x):
            x = K.max(x, axis = 2, keepdims = True)
            return x
        
        def reduce_dim_shape(input_shape):
            shape = list(input_shape)
            shape[-1] = 1
            print ("reduced_shape", shape)
            return tuple(shape)
        
        earlyStopping = EarlyStopping(monitor='loss', patience=2, verbose=0, mode='auto')
        reduced = Lambda(reduce_dim, output_shape = reduce_dim_shape)(merged)
        model = Model(inputs=[x,y_order], outputs=merged)
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
        
        model.fit([self.x_train, self.y_train_order], self.y_train, batch_size = self.batch_size,epochs=self.epoch, callbacks = [earlyStopping], validation_split = 0.2, shuffle = True)
        #for layer in model.layers:
        #    weights = layer.get_weights()
        #    print (weights)
        #for layer in model.layers:
        #        print (np.shape(layer.get_weights()))            
        #validation_data=([self.x_train,self.y_train_order],self.y_train))
        #score = model.evaluate([self.x_test, self.y_test_order], self.y_test, batch_size= self.batch_size)
        #print (score)
        # pirint (model.predict([self.x_train, self.y_train_order]))
        
        