from keras.models import Model
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
class DKTnet():
    def __init__(self, input_dim, input_dim_order,batch_size, epoch,
        x_train, y_train, y_train_order,
        x_test, y_test, y_test_order):

        self.input_dim = int(input_dim)
        self.input_dim_order = int(input_dim_order)
        self.batch_size = int(batch_size)
        self.epoch = int(epoch)

        self.x_train = x_train
        self.y_train = y_train
        self.y_train_order = y_train_order

        self.x_test = x_test
        self.y_test = y_test
        self.y_test_order = y_test_order



    def build(self):
        x = Input(batch_shape=(None,None, self.input_dim), name='x')
        y_order = Input(shape = (None,self.input_dim_order), name = 'y_order')
        masked = (Masking(mask_value= -1,input_shape=(None,None,self.input_dim)))(x)
        lstm_out = LSTM(self.input_dim_order, input_shape = (None, None, self.input_dim), return_sequences = True)(masked)
        #merged = lstm_out
        #softmaxed = Dense(124, activation = "softmax")(lstm_out)
        #pdb.set_trace()
        merged=merge([lstm_out,y_order],mode='mul')

        def reduce_dim(x):
            x = K.max(x,axis = 2, keepdims = True)
            return x
        def reduce_dim_shape(input_shape):
            shape = list(input_shape)
            shape[-1] = 1
            print "reduced_shape",shape
            return tuple(shape)

        reduced = Lambda(reduce_dim,output_shape = reduce_dim_shape)(merged)
        model = Model(inputs=[x,y_order],outputs=reduced)
        histories = my_callbacks.Histories()
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
        model.fit([self.x_train, self.y_train_order], self.y_train, batch_size = self.batch_size,epochs=self.epoch, callbacks =
                [histories],
        validation_split = 0.2, shuffle = True)
        #validation_data=([self.x_train,self.y_train_order],self.y_train))
        score = model.evaluate([self.x_test, self.y_test_order], self.y_test, batch_size= self.batch_size)

        # model.predict


