import tensorflow as tf
import keras.backend as K


def focal_loss(gamma):
    """

    :param gamma:
    :return:
    """
    def focLoss(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -K.sum(K.pow(1. - pt_1, gamma) * K.log(pt_1)) - K.sum(K.pow(pt_0, gamma) * K.log(1. - pt_0))

    return focLoss


def reduced_focal_loss():
    """
    #TODO add reduced focal loss
    :return:
    """

    return




def multi_weighted_logloss(class_weight):
    """
    #multi logloss
    classes = ['mu', 'el', 'tau', 'jet', 'gentau']

    """
    def mywloss(y_true, y_pred):
        yc = tf.clip_by_value(y_pred, 1e-15, 1 - 1e-15)
        loss = -(tf.reduce_mean(tf.reduce_mean(y_true * tf.log(yc), axis=0) / class_weight))
        return loss

    return mywloss

