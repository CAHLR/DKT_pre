
import keras
from sklearn.metrics import roc_auc_score
from sklearn.metrics import auc
import numpy as np
import pdb
class Histories(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.aucs = []
        self.losses = []

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        self.losses.append(logs.get('loss'))
        y_pred = self.model.predict(self.validation_data[0:2])
        yp = []
        yt = []
        print type(yt)
        print type(yp)

        for i in xrange(len(self.validation_data[1])):
            for j in xrange(len(self.validation_data[1][i])):
                if(self.validation_data[2][i][j][0]) == -1:
                    break
                else:
                    yp.append([y_pred[i][j][0]])
                    yt.append([self.validation_data[2][i][j][0]])
                #if self.validation_data[1][i][j][0] == -1:
                    #continue
                #else:
                    #amax = np.argmax(self.validation_data[1][i][j])
                    #yp.append([y_pred[i][j][amax]])
                    #yt.append([self.validation_data[2][i][j][amax]])
        #pdb.set_trace()
        tmp_auc = roc_auc_score(yt,yp)
        print "yt",len(yt)
        print "yp",len(yp)
        #auc = roc_auc_score(yt[0], yp[0])
        self.aucs.append(tmp_auc)
        print 'val-loss',logs.get('loss'), ' val-auc: ',tmp_auc
        print '\n'

        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return

