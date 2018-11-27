import tensorflow as tf
import numpy as np

from nn_common import BasicNN
from utils import *
import gc
import time


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
        self.coef_hardloss = tf.placeholder(tf.float32)
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
        # self.loss = loss_soft*self.coef_softloss*tf.square(self.temperature) + loss_standard*(1-self.coef_softloss) # TODO: back
        self.loss = loss_soft*self.coef_softloss + loss_standard*self.coef_hardloss # TODO: back

        self.opt = opt
        self.train_op = self.opt.minimize(self.loss)
        self.metrics = metrics
        if metrics is not None:
            for m_name in metrics:
                self.his_metrics_train[m_name] = []
                self.his_metrics_train_epoch[m_name] = []
                self.his_metrics_val[m_name] = []

        self.initialize_metric_tensor()
        

        
        
    def soft_train(self, X, y, y_soft, temperature, coef_softloss, coef_hardloss, n_epochs, batch_size=None, val_set=None, display_steps=50, shuffle=True, earlystop_params={}): 
        
        check_available_device()
        # data_valid:list
        self.session.run(tf.global_variables_initializer())

        assert X.shape[0] == y.shape[0]
        n_samples = X.shape[0]

        if batch_size is None:
            batch_size = n_samples

        steps_per_epoch = int(n_samples//batch_size)
        counter = 0

        # earlystop
        if earlystop_params != {}:
            monitor = earlystop_params['monitor']
            patience = earlystop_params['patience']
            earlystop = {
            'max_metric':-np.inf, 
            'epoch_diff':0, 
            'stop':False
            }

        for epoch in range(1,n_epochs+1):

            # if shuffle:
            #     order = np.random.permutation(n_samples)
            #     X = X[order]
            #     y = y[order]
            #     y_soft = y_soft[order] # holyyyyyyyyyyyyyy

            if shuffle:
                order = np.random.permutation(n_samples)
            #     X = X[order]
            #     y = y[order]
            else:
                order = np.arange(n_samples)

            for step in range(0,steps_per_epoch): # n_sample=1000, batch_size=10, steps_per_epoch=100
                start_step = time.clock()

                t_cost = {} # compute each computation cost time
                
                # if step != steps_per_epoch-1: # last step
                #     X_batch = X[step*batch_size:(step+1)*batch_size]
                #     y_batch = y[step*batch_size:(step+1)*batch_size]
                #     y_soft_batch = y_soft[step*batch_size:(step+1)*batch_size]
                # else:
                #     X_batch = X[step*batch_size:]
                #     y_batch = y[step*batch_size:]
                #     y_soft_batch = y_soft[step*batch_size:]

                indices = order[step*batch_size:(step+1)*batch_size]
                X_batch = X[indices]
                y_batch = y[indices]
                y_soft_batch = y_soft[indices]
                
                # train
                start_train = time.clock()
                feed_dict = {
                    self.Xs:X_batch
                    , self.ys:y_batch
                    , self.y_soft:y_soft_batch, self.temperature:temperature
                    , self.coef_softloss:coef_softloss, self.coef_hardloss:coef_hardloss
                    }
                feed_train = {**feed_dict, **self.feed_dict_train}

                __, loss_train = self.session.run(
                    [self.train_op, self.loss]
                    , feed_dict=feed_train
                )
                t_cost['train_op + train_loss'] = time.clock()-start_train

                # start_loss = time.clock()
                # loss_train = self.session.run(self.loss,feed_dict=feed_train)
                # start_append = time.clock()
                # t_cost['loss_train'] = start_append-start_loss
                self.his_loss_train.append(loss_train)

                if self.metrics is not None:
                    start_metric = time.clock()
                    m = self.get_metrics(X_batch, y_batch)
                    t_cost['metric batch'] = time.clock()-start_metric
                    for m_name,m_value in m.items():
                        self.his_metrics_train[m_name].append(m_value)
                # if counter%display_steps==0 or (epoch==n_epochs and step==steps_per_epoch-1):
                if counter%display_steps==0 or (step==steps_per_epoch-1):
                    
                    print('Epoch',epoch,', step',step,', loss=',loss_train, end=' ')

                    if val_set is not None and step==steps_per_epoch-1: # TODO: X_val, y_val go first
                        X_val = val_set[0]
                        y_val = val_set[1]
                        y_val_soft = val_set[2]
                        start_loss_val = time.clock()
                        loss_val = self.session.run(self.loss,feed_dict={self.Xs:X_val, self.ys:y_val, self.y_soft:y_val_soft, self.temperature:temperature
                            , self.coef_softloss:coef_softloss, self.coef_hardloss:coef_hardloss})
                        start_m_val = time.clock()
                        t_cost['loss_val'] = start_m_val-start_loss_val
                        if self.metrics is not None:
                            m_val = self.get_metrics(X_val,y_val)
                            t_cost['metric_val'] = time.clock()-start_m_val
                        self.his_loss_val.append(loss_val)
                        
                        print('val_loss=',loss_val, end=' ')
                    
                    if self.metrics is not None: # metrics
                        # start_metric = time.clock()
                        # m = self.get_metrics(X_batch, y_batch)
                        # t_cost['metric batch'] = time.clock()-start_metric
                        for m_name,m_value in m.items():
                            # self.his_metrics_train[m_name].append(m_value)
                            print(',', m_name,'=',m_value, end=' ')
                            
                            if val_set is not None and step==steps_per_epoch-1:
                                print('val',m_name,'=',m_val[m_name],end=' ')
                                self.his_metrics_val[m_name].append(m_val[m_name])
                    print()
                    
                    if step==steps_per_epoch-1:
                        loss_train_epoch = np.mean(self.his_loss_train[-steps_per_epoch:])
                        self.his_loss_train_epoch.append(loss_train_epoch)
                        print('Epoch',epoch,'finished, loss=',loss_train_epoch, end=' ')
                        if val_set is not None:
                            print('val loss=',loss_val, end=' ')
                        if self.metrics is not None:
                            for m_name in m:
                                metrics_train_epoch = np.mean(self.his_metrics_train[m_name][-steps_per_epoch:])
                                self.his_metrics_train_epoch[m_name].append(metrics_train_epoch)
                                print(', ',m_name,'=',metrics_train_epoch, end=' ')
                                if val_set is not None:
                                    print('val',m_name,'=',m_val[m_name])
                        # print()

                        # early stop
                        if earlystop_params!={}:
                            
                            if monitor=='val_loss':
                                if val_set is not None:
                                    early_metric = -self.his_loss_val[-1]
                                else:
                                    print('Error: No val_set to get val_loss in earlystopping. ')
                            elif monitor=='loss':
                                early_metric = -self.his_loss_train_epoch[-1]
                            elif 'val_' in monitor: # metrics
                                if val_set is not None:
                                    m_name = monitor.split('val_')[1]
                                    early_metric = self.his_metrics_val[m_name][-1]
                                else:
                                    print('Error: No val_set to get',monitor,'in earlystopping. ')
                            else:
                                early_metric = self.his_metrics_train_epoch[m_name][-1]


                            if early_metric > earlystop['max_metric']:
                                earlystop['max_metric'] = early_metric
                                earlystop['epoch_diff'] = 0
                                print(monitor, 'improved. ')
                            else:
                                earlystop['epoch_diff'] += 1
                                print()

                            if earlystop['epoch_diff'] > patience:
                                print('Stop training,', monitor, "didn't improve more than", patience, 'epoch. ')
                                earlystop['stop'] = True
                                break

                    t_cost['whole'] = time.clock()-start_step
                    # t_cost['display_whole'] = time.clock()-start_loss
                    # print_obj(t_cost,'t_cost')

                
                # gc.collect()
                counter += 1

            if earlystop_params!={} and earlystop['stop']:
                break








