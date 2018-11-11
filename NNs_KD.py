import tensorflow as tf
import numpy as np

from NNs import BasicNN



class TeacherNN(BasicNN):

    def __init__(self, input_dims, output_dims, dtype_X, dtype_y, session=None, ckpt_dir=None, ckpt_file=None, log_dir='logs'
        , temperature=1, model_type=None):

        super(BasicNN, self).__init__(input_dims, output_dims, dtype_X, dtype_y, session=session, ckpt_dir=ckpt_dir, ckpt_file=ckpt_file, log_dir=log_dir)

        self.temperature = temperature
        self.model_type = model_type












