# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 16:41:27 2020

@author: UC
"""

from keras import backend as K
from keras.losses import categorical_crossentropy
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
#                                reduction_indices=len(output.get_shape()) - 1)
#     else:
#         return tf.nn.softmax_cross_entropy_with_logits(labels=target,
#                                                        logits=output)
   
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
    from keras.losses import categorical_crossentropy
    mask = 1 - gt[:, :, 0]
    return categorical_crossentropy(gt, pr) * mask

# Custom summed losses
def jaccard_crossentropy(out, tar):
    return categorical_crossentropy(out, tar) + jaccard_distance(out, tar)

def masked_jaccard_crossentropy(out, tar):
    return masked_categorical_crossentropy(out, tar) + masked_jaccard_distance(out, tar)

# Metrics
def masked_categorical_accuracy(y_true, y_pred):
    from keras.metrics import categorical_accuracy
    mask = 1 - y_true[:, :, 0]
    return categorical_accuracy(y_true, y_pred) * mask