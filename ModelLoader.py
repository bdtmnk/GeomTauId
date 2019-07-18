#!/usr/bin/python
# Written by Owen Colegrove
# This class is responsible for building the deep-learning model in keras
# For a better understanding of the code structure, refer to "Trainer.py"
import keras
from keras.models import Model
from keras.layers import *
from keras.layers.advanced_activations import LeakyReLU
from keras.callbacks import Callback
from keras.optimizers import Adam
from keras.utils import plot_model
import tensorflow as tf

def auc(y_true, y_pred):
    auc = tf.metrics.auc(y_true, y_pred)[1]
    K.get_session().run(tf.local_variables_initializer())
    return auc

class ModelLoader:
  def __init__(self, inshape):
    self.inshape = inshape
      
  def load(self):
    modelinshape = Input(self.inshape)
    model = Conv1D(1024,kernel_size=4,padding="same")(modelinshape)
    model = LeakyReLU(alpha=0.01)(model)

    model = Conv1D(1024,kernel_size=4,padding="same")(model)
    model = LeakyReLU(alpha=0.01)(model)

    model = BatchNormalization()(model)

    model = Conv1D(512,kernel_size=4,padding="same")(model)
    model = LeakyReLU(alpha=0.01)(model)

    model = Conv1D(512,kernel_size=4,padding="same")(model)
    model = MaxPooling1D()(model)
    model = LeakyReLU(alpha=0.01)(model)

    model = BatchNormalization()(model)

    model = Conv1D(256,kernel_size=3,padding="same")(model)
    model = LeakyReLU(alpha=0.01)(model)

    model = Conv1D(256,kernel_size=3,padding="same")(model)
    model = MaxPooling1D()(model)
    model = LeakyReLU(alpha=0.01)(model)

    model = BatchNormalization()(model)

    model = Conv1D(64,kernel_size=3,padding="same")(model)
    model = LeakyReLU(alpha=0.01)(model)

    model = Conv1D(64,kernel_size=3,padding="same")(model)
    model = MaxPooling1D()(model)
    model = LeakyReLU(alpha=0.01)(model)

    model = BatchNormalization()(model)

    model = Conv1D(16,kernel_size=2,padding="same")(model)
    model = LeakyReLU(alpha=0.01)(model)

    model = Conv1D(16,kernel_size=2,padding="same")(model)
    model = MaxPooling1D()(model)
    model = LeakyReLU(alpha=0.01)(model)

    model = Flatten()(model)

    model = Dense(2048,activation=None)(model)
    model = LeakyReLU(alpha=0.01)(model)

    model = Dense(256,activation=None)(model)
    model = LeakyReLU(alpha=0.01)(model)

    model = Dense(64,activation=None)(model)
    model = LeakyReLU(alpha=0.01)(model)

    model = Dense(1,activation='sigmoid')(model)

    final = Model(inputs=modelinshape,outputs=model)
    csv_logger = keras.callbacks.CSVLogger('CNNMCValid.log')
    lr = 0.001
    opt = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    final.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy',auc])
    self.final = final
    plot_model(final, 'conv1d.svg')
    return final

  def load_multiclass(self, ouput_class=3,  loss='categorical_crossentropy',weights=None):
    """

    :return:
    """
    modelinshape = Input(self.inshape)
    model = Conv1D(1024, kernel_size=4, padding="same")(modelinshape)
    model = LeakyReLU(alpha=0.01)(model)

    model = Conv1D(1024, kernel_size=4, padding="same")(modelinshape)
    model = LeakyReLU(alpha=0.01)(model)

    model = Conv1D(512, kernel_size=4, padding="same")(model)
    model = LeakyReLU(alpha=0.01)(model)

    model = Conv1D(512, kernel_size=4, padding="same")(model)
    model = MaxPooling1D()(model)
    model = LeakyReLU(alpha=0.01)(model)

    model = Conv1D(256, kernel_size=3, padding="same")(model)
    model = LeakyReLU(alpha=0.01)(model)

    model = Conv1D(256, kernel_size=3, padding="same")(model)
    model = MaxPooling1D()(model)
    model = LeakyReLU(alpha=0.01)(model)

    model = Conv1D(64, kernel_size=3, padding="same")(model)
    model = LeakyReLU(alpha=0.01)(model)

    model = Conv1D(64, kernel_size=3, padding="same")(model)
    model = MaxPooling1D()(model)
    model = LeakyReLU(alpha=0.01)(model)

    model = Conv1D(16, kernel_size=2, padding="same")(model)
    model = LeakyReLU(alpha=0.01)(model)

    model = Conv1D(16, kernel_size=2, padding="same")(model)
    model = MaxPooling1D()(model)
    model = LeakyReLU(alpha=0.01)(model)

    model = Flatten()(model)

    model = Dense(2048, activation=None)(model)
    model = LeakyReLU(alpha=0.01)(model)

    model = Dense(256, activation=None)(model)
    model = LeakyReLU(alpha=0.01)(model)

    model = Dense(64, activation=None)(model)
    model = LeakyReLU(alpha=0.01)(model)

    model = Dense(ouput_class, activation='softmax')(model)

    final = Model(inputs=modelinshape, outputs=model)
    csv_logger = keras.callbacks.CSVLogger('CNNMCValid.log')
    lr = 0.0001
    opt = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    final.compile(loss=loss, optimizer=opt, metrics=['accuracy'])#, metrics=['accuracy'])
    plot_model(final, 'conv1d.png')
    return final

  def set_loss(self, loss):
    ## Set loss for the
    lr = 0.0001
    opt = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    self.final.compile(loss=loss, optimizer=opt, metrics=['accuracy',auc])
    return self.final
