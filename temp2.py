from keras.models import Model
from keras.layers import Input, Dense, Dropout 
from keras.layers import Embedding
from keras.layers import LSTM
from keras.layers import merge
from theano import tensor as T
import numpy as np

batch_size = 64
input_dim = 1000
input_dim_order =  500
x = Input(batch_shape=(None,None, input_dim), name='x')
y_order = Input(shape = (None,500), name = 'y_order') 

lstm_out = LSTM(500, input_shape = (None, None, input_dim), return_sequences = True)(x)

merged=merge([lstm_out,y_order],mode='mul')
model = Model(inputs = [x, y_order], outputs = merged)
model.compile( optimizer = 'rmsprop', 
                loss = 'categorical_crossentropy',
                metrics=['accuracy'])# loss_weights?

model.fit([self.x_train, self.y_train_order], self.y_train, batch_size=16, epochs=10)
score = model.evaluate([self.x_test, self.y_test_order], self.y_test, batch_size=16)

# model.predict
