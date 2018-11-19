import numpy as np
import os
import tensorflow as tf
import random
from tensorflow.python.client import device_lib

def create_session(gpu_id='0', pp_mem_frac=None):

    tf.reset_default_graph()
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id # can multiple?
    with tf.device('/gpu:' + gpu_id):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        if pp_mem_frac is not None:
            config.gpu_options.per_process_gpu_memory_fraction=pp_mem_frac
        session = tf.Session(config = config)
    return session
    
def close_session(session):
    session.close()

def check_available_device():
    print(device_lib.list_local_devices())

def set_rand_seed(seed=None): # TODO: keras, theano and so forth
    if seed is None:
        # seed = int(os.getenv("SEED", 12))
        pass
    tf.set_random_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def get_tensor_shape(tensor):
    return tensor.get_shape().as_list(), tensor.get_shape().num_elements()

def print_obj(obj, obj_str): # support ndarray, Tensor, dict, iterable
    # exec('global '+obj_str)
    # exec('obj = '+obj_str)
    obj_type = type(obj)
    print(obj_str
        , obj_type
        , end = ' '
        )
    if obj_type == np.ndarray:
        print(obj.shape)

    elif tf.contrib.framework.is_tensor(obj):
        obj_shape = obj.get_shape().as_list() # get_shape().num_elements() can get sum
        print(obj_shape)
    elif isinstance(obj, dict):
        print()
        for key, content in obj.items():
            print(key, ':', content)
    else:
        try:
            iterator = iter(obj)
        except TypeError:
            # not iterable
            print(obj)
        else:
            # iterable
            print(len(obj))

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]



