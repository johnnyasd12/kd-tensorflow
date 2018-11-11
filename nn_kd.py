import tensorflow as tf
import numpy as np

from nn_common import BasicNN



class TeacherNN(BasicNN):

    def __init__(self, input_dims, output_dims, dtype_X, dtype_y, session=None, ckpt_dir=None, ckpt_file=None, log_dir='logs'
        , temperature=1, model_type=None):

        super(TeacherNN, self).__init__(input_dims, output_dims, dtype_X, dtype_y, session=session, ckpt_dir=ckpt_dir, ckpt_file=ckpt_file, log_dir=log_dir)

        self.temperature = tf.placeholder(tf.float32)
        self.model_type = model_type

        self.logits_with_T = None
        self.softened_softmax = None

    def predict_with_t(self, X, temperature):

        self.logits_with_T = self.logits / self.temperature
        self.softened_softmax = tf.nn.softmax(self.logits_with_T)
        prediction = self.session.run(self.softened_softmax, feed_dict={self.Xs:X, self.temperature:temperature})
        return prediction


class StudentNN(TeacherNN): # TODO: TeacherNN?
    
    def __init__(self, input_dims, output_dims, dtype_X, dtype_y, session=None, ckpt_dir=None, ckpt_file=None, log_dir='logs', temperature=1, model_type=None):

        super(StudentNN, self).__init__(input_dims, output_dims, dtype_X, dtype_y, session=session, ckpt_dir=ckpt_dir, ckpt_file=ckpt_file, log_dir=log_dir, temperature=temperature, model_type=model_type)

        self.loss_total = None
        self.loss_standard = None
        self.loss_soft = None








