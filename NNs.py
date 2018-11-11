import tensorflow as tf
import numpy as np
from pprint import pprint
from param_collection import ParamCollection

import random
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '1'
import matplotlib.pyplot as plt
from utils import *

class BasicNN(object):

    def __init__(self, input_dims, output_dims, dtype_X, dtype_y, session=None):
#         input_dims = topology[0] # TODO: after modified into function
        # Layers in network.
#         L = len(topology) - 1
        
        #         self.L = L # n_layers except input layer
#         self.topology = topology
#         self.layers_func = layers_func # TODO: want to modify to FUNCTION
        self.L = 0 # n_layers except input layer
        self.topology = [input_dims] # output dims of each layer
        self.activations = []
        self.layer_funcs = []
        # tensorflow
        if session is None:
            self.session = create_session(gpu_id='0')
        else:
            self.session = session
        self.dtype_X = dtype_X
        self.dtype_y = dtype_y
        # TODO: loop input_dims for initializing shape
        with tf.name_scope('inputs'):
            self.Xs = tf.placeholder(dtype_X,shape=[None, input_dims]) # output of input layer
            self.ys = tf.placeholder(dtype_y,shape=[None, output_dims])
        # below are Tensor
        self.W = [] # weights in each layer
        self.b =[] # biases in each layer
        self.h = [self.Xs] # activation output in each layer
        self.params = [] # store all Ws and bs, will be modified when W and b is trained i think
        self.logits = None # neurons before input to final activation
        self.prediction = None
        self.loss = None # output loss
        # above are Tensor
        self.opt = None
        self.train_op = None # opt.minimize(self.loss)
        self.metrics = None # metric names
        # loss, metrics history
        self.his_loss_train = []
        self.his_loss_val = []
        self.his_metrics_train = {}
        self.his_metrics_val = {}
        # param collection
        self.pc = None
        
        
    def weight_fc(self, shape, stddev=0.1, initial=None):
        if initial is None:
            initial = tf.truncated_normal(shape,stddev=stddev,dtype=self.dtype_X)
            initial = tf.Variable(initial)
            # initial = tf.Variable(tf.random_normal(shape), dtype=self.dtype_X)
        return initial

    def bias_fc(self, shape, init_bias=0.1, initial=None):
        if initial is None:
            initial = tf.constant(init_bias,shape=shape,dtype=self.dtype_X)
            initial = tf.Variable(initial)
            # initial = tf.Variable(tf.zeros(shape) + 0.1, dtype=self.dtype_X)
        return initial
    
    def add_fc(
        self
#         , inputs
#         , in_size
        , out_size, activation_func=None
        # , output_layer=False
        , initial=None):
        
#         Weights = tf.Variable(tf.random_normal([in_size,out_size]))
#         biases = tf.Variable(tf.zeros([1,out_size]) + 0.1)
        inputs = self.h[-1] # last layer output as input
        shape_inputs = inputs.get_shape().as_list()
        self.L = self.L + 1
        print('Layer',self.L,': FC, input shape =',shape_inputs,', out_size =',out_size)
        in_size = shape_inputs[1]#self.session.run(tf.shape(inputs))[1] # TODO: get input shape = [1,out_size]
        shape_W = [in_size,out_size]
        shape_b = [1,out_size]
#         print('FC_layer, shape_W =',shape_W,', shape_b =',shape_b)
        
        Weights = self.weight_fc(shape_W)
        biases = self.bias_fc(shape_b)
        WX_b = tf.matmul(inputs, Weights) + biases
        if activation_func is None:
            out = WX_b
        else:
            out = activation_func(WX_b)

        self.topology.append(out_size)
        self.activations.append(activation_func)
        self.layer_funcs.append(self.add_fc)
        self.W.append(Weights)
        self.b.append(biases)
        self.h.append(out)
        # if output_layer:
        self.logits = WX_b
        self.prediction = self.h[-1]
        
        # for ParamCollection
        self.params.append(self.W[-1])
        self.params.append(self.b[-1])
#         self.pc = ParamCollection(self.session, params) # TODO: watch this

    def compile_nn(self, loss, opt, metrics=None):
        # metrics:list
        
        self.loss = loss
        self.opt = opt
        self.train_op = self.opt.minimize(self.loss)
        self.metrics = metrics
        if metrics is not None:
            for m_name in metrics:
                self.his_metrics_train[m_name] = []
                self.his_metrics_val[m_name] = []
        self.pc = ParamCollection(self.session, self.params)
        
        self.session.run(tf.global_variables_initializer())
        
    def train(self, X, y, n_epochs, batch_size=None, val_set=None, display_steps=50, shuffle=True): 
        # data_valid:list
        assert X.shape[0] == y.shape[0]
        n_samples = X.shape[0]
        if batch_size is None:
            batch_size = n_samples
        steps_per_epoch = int(n_samples/batch_size)
        counter = 0
        for epoch in range(1,n_epochs+1):
            if shuffle:
                order = np.random.permutation(n_samples)
                X = X[order]
                y = y[order]
            for step in range(0,steps_per_epoch): # n_sample=1000, batch_size=10, steps_per_epoch=100
                if step != steps_per_epoch-1: # last step
                    X_batch = X[step*batch_size:(step+1)*batch_size]
                    y_batch = y[step*batch_size:(step+1)*batch_size]
                else:
                    X_batch = X[step*batch_size:]
                    y_batch = y[step*batch_size:]
                
                self.session.run(
                    self.train_op
                    , feed_dict={self.Xs:X_batch, self.ys:y_batch}
                )
                if counter%display_steps==0 or (epoch==n_epochs and step==steps_per_epoch-1):
                    
                    loss_train = self.session.run(self.loss,feed_dict={self.Xs:X_batch, self.ys:y_batch})
                    self.his_loss_train.append(loss_train)
                    print('Epoch',epoch,', step',step,', loss=',loss_train, end=' ')
                    
                    if self.metrics is not None:
#                         y_pred_batch = self.session.run(self.prediction,feed_dict={self.Xs:X_batch})
                        m = self.get_metrics(X_batch, y_batch)
                        for m_name,m_value in m.items():
                            print(m_name,'=',m_value, end=' ')
                            self.his_metrics_train[m_name].append(m_value)

                    if val_set is not None:
                        X_val = val_set[0]
                        y_val = val_set[1]
                        loss_val = self.session.run(self.loss,feed_dict={self.Xs:X_val,self.ys:y_val})
                        self.his_loss_val.append(loss_val)
                        print(', val_loss=',loss_val, end=' ')
                        if self.metrics is not None:
#                             y_pred_val = self.session.run(self.prediction,feed_dict={self.Xs:X_val})
                            m_val = self.get_metrics(X_val,y_val)
                            for m_name,m_value in m_val.items():
                                print('val',m_name,'=',m_value,end=' ')
                                self.his_metrics_val[m_name].append(m_value)
                    print()
                    
                counter += 1
                
    
    def predict(self, X):
        return self.session.run(self.prediction,feed_dict={self.Xs:X})

    def get_metrics(self, X, y):
        func = {
            'acc':self.compute_accuracy
        }
        dict_metrics = {}
        for m_name in self.metrics:
            if isinstance(m_name,str):
                dict_metrics[m_name] = func[m_name](X,y)
            else: # TODO: gogogo
                pass
        return dict_metrics
    def compute_accuracy(self, X, y): # input array
        correct_prediction = tf.equal(tf.argmax(self.prediction,axis=1), tf.argmax(self.ys,axis=1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, self.dtype_X))
        result = self.session.run(accuracy, feed_dict={self.Xs: X, self.ys: y})
        return result
    
    def plt_loss(self, title='loss'):
        print('Plotting loss...')
        loss_t = self.his_loss_train
        loss_v = self.his_loss_val
        plt.title(title)
        plt.plot(loss_t, label='training loss')
        if len(self.his_loss_val) != 0:
            plt.plot(loss_v, label='validation loss')
        plt.legend()
        plt.show()
    
    def plt_metrics(self):
        print('Plotting metrics...')
        if self.metrics is None:
            print('no metrics to plot')
        else:
            for m_name in self.metrics:
                plt.title('metrics: '+m_name)
                plt.plot(self.his_metrics_train[m_name], label='training '+m_name)
                plt.plot(self.his_metrics_val[m_name], label='validation '+m_name)
                plt.legend()
                plt.show()
        
    def save_model(self): # TODO
        pass
    def load_model(self): # TODO
        pass

























