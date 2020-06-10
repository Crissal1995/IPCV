# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 16:41:27 2020

@author: UC
"""

from keras import backend as K
from keras.losses import categorical_crossentropy
import tensorflow as tf
import sys
#import ridurre

#from keras import metrics
# import tensorflow as tf

# class MeanIoU(metrics.MeanIoU):
#     def __call__(self, y_true, y_pred, sample_weight=None):
#         y_pred = tf.argmax(y_pred, axis=-1)
#         return super().__call__(y_true, y_pred, sample_weight=sample_weight)

# def categorical_crossentropy(output, target, from_logits=False):
#     """Categorical crossentropy between an output tensor and a target tensor.
#     # Arguments
#         output: A tensor resulting from a softmax
#             (unless `from_logits` is True, in which
#             case `output` is expected to be the logits).
#         target: A tensor of the same shape as `output`.
#         from_logits: Boolean, whether `output` is the
#             result of a softmax, or is a tensor of logits.
#     # Returns
#         Output tensor.
#     """
#     # Note: tf.nn.softmax_cross_entropy_with_logits
#     # expects logits, Keras expects probabilities.
#     if not from_logits:
#         # scale preds so that the class probas of each sample sum to 1
#         output /= tf.reduce_sum(output,
#                                 reduction_indices=len(output.get_shape()) - 1,
#                                 keep_dims=True)
#         # manual computation of crossentropy
#         epsilon = _to_tensor(_EPSILON, output.dtype.base_dtype)
#         output = tf.clip_by_value(output, epsilon, 1. - epsilon)
#         return - tf.reduce_sum(target * tf.log(output),
#                                 reduction_indices=len(output.get_shape()) - 1)
#     else:
#         return tf.nn.softmax_cross_entropy_with_logits(labels=target,
#                                                         logits=output)
   
# Losses 
def jaccard_distance(y_true, y_pred, smooth=100):
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return (1 - jac) * smooth

def masked_jaccard_distance(y_true, y_pred, smooth=100):
    mask = 1 - y_true[:, :, 0]
    return jaccard_distance(y_true, y_pred) * mask

def masked_categorical_crossentropy(gt, pr):
    print(gt)
    print(pr)
    from keras.losses import categorical_crossentropy
    mask = 1 - gt[:, :, 0]
    return categorical_crossentropy(gt, pr) * mask

def generate_onehot(a):
    # an = [[[1,3,5],[2,1,1]],[[3,4,1],[4,2,3]]]
    # a = tf.convert_to_tensor(an,dtype=tf.float32)
    # tf.print(a)
    max_elems = tf.reduce_max(a,2,keepdims=True)
    tf.print(max_elems)
    nonpos = tf.subtract(a,max_elems)
    tf.print(nonpos)
    zero = tf.constant(0, dtype=tf.float32)
    one = tf.constant(1, dtype=tf.float32)
    condition = tf.equal(nonpos,zero)
    onehot = tf.where(condition, one, zero)
    tf.print(onehot)
    return onehot

import numpy as np
def mild_categorical_crossentropy(gt, pr):
    from keras.losses import categorical_crossentropy
    
    gt0 = tf.zeros(shape=tf.shape(gt[:,:,0]))
    gt0 = gt0[:,:,np.newaxis]
    
    # Metodo equiparante
    # gt_0 = tf.divide(gt[:,:,0],37)
    # gtx = tf.repeat(gt_0[:,:,np.newaxis],37,axis=2)
    # gt1 = tf.add(gt[:,:,1:],gtx) 
    # Metodo meritocratico
    gtonehot = generate_onehot(pr[:,:,1:])
    gtmask = tf.repeat(gt[:,:,0,np.newaxis],37,axis=2)
    gtmult = tf.multiply(gtonehot,gtmask)
    gt1 = tf.add(gt[:,:,1:],gtmult)
    gtf = tf.concat([gt0,gt1],2)
    
    
    #if tf.is_nan(K.any(gt)):
    # zero = tf.constant(0, dtype=tf.float32)
    # one = tf.constant(1, dtype=tf.float32)
    # whereGT = tf.equal(gt, zero)
    # #whereGT = tf.dtypes.cast(whereGT, tf.float32)
    # wherePR = tf.not_equal(gt, zero)
    # #wherePR = tf.dtypes.cast(wherePR, tf.float32)
    # gt = tf.cond(tf.logical_and(whereGT,wherePR), lambda: tf.reduce_sum([[gt],[pr]],0), lambda: gt)
        
        # for idx_3, (yt_3, yp_3) in enumerate(zip(gt.numpy(),pr.numpy())):
        #     for idx_2, (yt_2, yp_2) in enumerate(zip(yt_3,yp_3)):
        #         for idx_1, (yt_1, yp_1) in enumerate(zip(yt_2,yp_2)):
        #             for idx_0, (yt_0, yp_0) in enumerate(zip(yt_1,yp_1)):
        #                 if yt_0 == 0 and yp_0 != 0:
        #                     gt[idx_0,idx_1,idx_2,idx_3] = yp_0
                    
    return categorical_crossentropy(gtf, pr)

# Custom summed losses
def jaccard_crossentropy(out, tar):
    return categorical_crossentropy(out, tar) + jaccard_distance(out, tar)

def masked_jaccard_crossentropy(out, tar):
    return masked_categorical_crossentropy(out, tar) + masked_jaccard_distance(out, tar)

# Metrics
# def mild_categorical_accuracy(y_true, y_pred):
#     return K.cast(K.equal(K.argmax(y_true, axis=-1),
#                           K.argmax(y_pred, axis=-1)),
#                   K.floatx())

def masked_categorical_accuracy(y_true, y_pred):
    from keras.metrics import categorical_accuracy
    mask = 1 - y_true[:, :, 0]
    return categorical_accuracy(y_true, y_pred) * mask