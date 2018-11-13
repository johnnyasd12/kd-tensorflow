import tensorflow as tf
from nn_kd import *
from layers import *


def build_big_model(input_dims, output_dims, session):

    bigmodel = SoftenedNN(
        input_dims=input_dims
        , output_dims=output_dims
        , session=session
        , dtype_X=tf.float32
        , dtype_y=tf.float32)

    bigmodel.add_layer(
        FC(inputs=bigmodel.h[-1],out_dims=64, activation_fn=tf.nn.relu))
    # bigmodel.add_layer(
    #     Dropout(inputs=bigmodel.h[-1], keep_prob=0.5))
    # bigmodel.add_layer(
    #     FC(inputs=bigmodel.h[-1],out_dims=1200, activation_fn=tf.nn.relu))
    # bigmodel.add_layer(
    #     Dropout(inputs=bigmodel.h[-1], keep_prob=0.5))
    bigmodel.add_layer(
        FC(inputs=bigmodel.h[-1],out_dims=output_dims, activation_fn=tf.nn.softmax))

    bigmodel.compile_nn(
        loss=tf.losses.softmax_cross_entropy(bigmodel.ys,bigmodel.logits)
        , opt=tf.train.AdamOptimizer(learning_rate=1e-3)
        , metrics = ['acc'])

    return bigmodel

def build_small_model(input_dims, output_dims, session, is_student):

    smallmodel = StudentNN(
        input_dims=input_dims
        , output_dims=output_dims
        , session=session
        , dtype_X=tf.float32
        , dtype_y=tf.float32)

    smallmodel.add_layer(
        FC(inputs=smallmodel.h[-1],out_dims=5,activation_fn=tf.nn.relu))
    smallmodel.add_layer(
        FC(inputs=smallmodel.h[-1],out_dims=output_dims, activation_fn=tf.nn.softmax))

    if is_student:
        smallmodel.compile_student(
            loss_standard=tf.losses.softmax_cross_entropy(smallmodel.ys,smallmodel.logits)
            , opt=tf.train.AdamOptimizer(learning_rate=1e-3) #TODO: higher learning rate
            # , opt=tf.train.AdagradOptimizer(learning_rate=1e-2)
            , metrics = ['acc']) # BUGFIX: logits_with_T still None currently
    else:
        smallmodel.compile_nn(
            loss=tf.losses.softmax_cross_entropy(smallmodel.ys,smallmodel.logits)
            , opt=tf.train.AdamOptimizer(learning_rate=1e-3)
            , metrics = ['acc']) # BUGFIX: logits_with_T still None currently
    return smallmodel


















