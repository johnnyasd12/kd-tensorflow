import numpy as np
import os
import tensorflow as tf
import random


def create_session(gpu_id='0', pp_mem_frac=None):
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

def set_rand_seed(seed=None): # TODO: keras, theano and so forth
    if seed is None:
        seed = int(os.getenv("SEED", 12))
    tf.set_random_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def print_obj(obj_str):
    exec('global '+obj_str)
    exec('obj = '+obj_str)
    obj_type = type(obj)
    print(obj_str
        , obj_type
        , end = ' '
        )
    if obj_type == np.ndarray:
        print(obj.shape)
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



