import tensorflow as tf
import numpy as np

from nn_common import BasicNN
from utils import *
import gc


class SoftenedNN(BasicNN):

    def __init__(self, input_dims, output_dims, dtype_X, dtype_y, session=None, ckpt_dir=None, ckpt_file=None, log_dir='logs', model_type=None):

        super(SoftenedNN, self).__init__(input_dims, output_dims, dtype_X, dtype_y, session=session, ckpt_dir=ckpt_dir, ckpt_file=ckpt_file, log_dir=log_dir)

        self.temperature = tf.placeholder(tf.float32)
        self.model_type = model_type

        self.logits_with_T = None
        self.softened_prediction = None

    def compile_nn(self, loss, opt, metrics=None):

        super(SoftenedNN, self).compile_nn(loss, opt, metrics)
        self.logits_with_T = self.logits / self.temperature
        # print_obj(self.logits,'self.logits')
        # print_obj(self.logits_with_T,'self.logits_with_T')
        self.softened_prediction = tf.nn.softmax(self.logits_with_T)
        # BUGFIX: session.global

    def predict_softened(self, X, temperature):

        prediction = self.session.run(self.softened_prediction, feed_dict={self.Xs:X, self.temperature:temperature})
        return prediction


class StudentNN(SoftenedNN): 
# class StudentNN(BasicNN):
    
    def __init__(self, input_dims, output_dims, dtype_X, dtype_y, session=None, ckpt_dir=None, ckpt_file=None, log_dir='logs', model_type=None):

        # super(StudentNN, self).__init__(input_dims, output_dims, dtype_X, dtype_y, session=session, ckpt_dir=ckpt_dir, ckpt_file=ckpt_file, log_dir=log_dir)
        # super(SoftenedNN, self).__init__(input_dims, output_dims, dtype_X, dtype_y, session=session, ckpt_dir=ckpt_dir, ckpt_file=ckpt_file, log_dir=log_dir)
        # self.temperature = tf.placeholder(tf.float32)
        # self.model_type = model_type
        # self.logits_with_T = None
        # self.softened_prediction = None

        super(StudentNN, self).__init__(input_dims, output_dims, dtype_X, dtype_y, session=session, ckpt_dir=ckpt_dir, ckpt_file=ckpt_file, log_dir=log_dir, model_type=model_type)

        self.y_soft = tf.placeholder(tf.float32, [None, output_dims], name='y_soft')
        self.coef_softloss = tf.placeholder(tf.float32)
        # self.loss_total = None
        self.loss_standard = None
        self.loss_soft = None

    def compile_student(self, loss_standard, opt, metrics=None): # TODO: written stuck is not proper
        
        # super(StudentNN, self).compile_nn(loss=loss, opt=opt, metrics=metrics) # BUGFIX: loss still standard while opt.minimize(loss)
        # super(SoftenedNN, self).compile_nn(loss_standard, opt, metrics)
        self.logits_with_T = tf.divide(self.logits, self.temperature) #self.logits / self.temperature
        self.softened_prediction = tf.nn.softmax(self.logits_with_T)

        # self.loss_standard = loss_standard
        loss_soft = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y_soft, logits=self.logits_with_T))
        self.loss_soft = loss_soft
        self.loss = loss_soft*self.coef_softloss + loss_standard # TODO: back

        self.opt = opt
        self.train_op = self.opt.minimize(self.loss)
        self.metrics = metrics
        if metrics is not None:
            for m_name in metrics:
                self.his_metrics_train[m_name] = []
                self.his_metrics_val[m_name] = []

        
        
    def soft_train(self, X, y, y_soft, temperature, coef_softloss, n_epochs, batch_size=None, val_set=None, display_steps=50, shuffle=True): 
        
        # data_valid:list
        self.session.run(tf.global_variables_initializer()) # BUGFIX: 
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
                y_soft = y_soft[order] # holyyyyyyyyyyyyyy

            for step in range(0,steps_per_epoch): # n_sample=1000, batch_size=10, steps_per_epoch=100

                if step != steps_per_epoch-1: # last step
                    X_batch = X[step*batch_size:(step+1)*batch_size]
                    y_batch = y[step*batch_size:(step+1)*batch_size]
                    y_soft_batch = y_soft[step*batch_size:(step+1)*batch_size]
                else:
                    X_batch = X[step*batch_size:]
                    y_batch = y[step*batch_size:]
                    y_soft_batch = y_soft[step*batch_size:]
                
                # train
                self.session.run(
                    self.train_op
                    , feed_dict={self.Xs:X_batch
                    , self.ys:y_batch
                    , self.y_soft:y_soft_batch, self.temperature:temperature, self.coef_softloss:coef_softloss}
                )

                if counter%display_steps==0 or (epoch==n_epochs and step==steps_per_epoch-1):
                # if counter%display_steps==0 or (step==steps_per_epoch-1):
                    
                    loss_train = self.session.run(self.loss,feed_dict={self.Xs:X_batch, self.ys:y_batch, self.y_soft:y_soft_batch, self.temperature:temperature, self.coef_softloss:coef_softloss})
                    self.his_loss_train.append(loss_train)
                    print('Epoch',epoch,', step',step,', loss=',loss_train, end=' ')
                    
                    if val_set is not None: # TODO: X_val, y_val go first
                        X_val = val_set[0]
                        y_val = val_set[1]
                        y_val_soft = val_set[2]
                        m_val = self.get_metrics(X_val,y_val)
                        loss_val = self.session.run(self.loss,feed_dict={self.Xs:X_val, self.ys:y_val, self.y_soft:y_val_soft, self.temperature:temperature, self.coef_softloss:coef_softloss})
                        self.his_loss_val.append(loss_val)
                        print('val_loss=',loss_val, end=' ')
                    
                    if self.metrics is not None: # metrics
                        m = self.get_metrics(X_batch, y_batch)
                        for m_name,m_value in m.items():
                            print(',', m_name,'=',m_value, end=' ')
                            self.his_metrics_train[m_name].append(m_value)
                        
                            if val_set is not None:
                                print('val',m_name,'=',m_val[m_name],end=' ')
                                self.his_metrics_val[m_name].append(m_val[m_name])
                    print()
                
                gc.collect()
                counter += 1








