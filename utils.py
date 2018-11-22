import numpy as np
import os
import tensorflow as tf
import random
from tensorflow.python.client import device_lib
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt

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

# plot and show sample # deprecated
def plot_mnist(X, y, y_soft=None, n_samples=6, round_show=4):

    n_data = X.shape[0]
    indices = np.random.choice(n_data,n_samples)
    for i in range(n_samples):
        sample = X[[indices[i]]]
        label = y[indices[i]]
        if y_soft is not None:
            label_soft = y_soft[indices[i]]
        plt.imshow(sample.reshape((28,28)), cmap='gray')
        plt.show()
        # pred = student.predict(sample)
        # pred_t = student.predict_softened(sample, temperature=temperature)
        print(label)
        print(label_soft)
        # print(np.round(pred,round_show))
        # print(np.round(pred_t,round_show))

def plot_confusion_matrix(y_true, y_pred):
    # This is called from print_test_accuracy() below.

    # cls_pred is an array of the predicted class-number for
    # all images in the test-set.

    # Get the number of classes
    num_classes = y_true.shape[1]
    
    # Get the confusion matrix using sklearn.
    cm = confusion_matrix(y_true=y_true,
                          y_pred=y_pred)

    # Print the confusion matrix as text.
    print(cm)

    # Plot the confusion matrix as an image.
    plt.matshow(cm)

    # Make various adjustments to the plot.
    plt.colorbar()
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, range(num_classes))
    plt.yticks(tick_marks, range(num_classes))
    plt.xlabel('Predicted')
    plt.ylabel('True')

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()

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



