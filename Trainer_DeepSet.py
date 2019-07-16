#!/usr/bin/python
# Written by Owen Colegrove
# Modified by leDidukh
'''
    The code below are scraps of actual code I used to train a deep neural network for object recognition using the CMS detector.  It greatly outperformed previous high-level techniques developed over many generations by high energy physicists.
    Raw detector information from is condensed down into a ~1 tb sample of 100M simulated particle decays with real and fake events.
    The events were generated and analyzed by us using a large cluster -- they are then stored into so called ROOT files we transfer them onto our private cluster.
    The dataset, loaded with EventFiller and Utilities.. ultimately things were modified s.t. training and event loading was done in parallel
    A deep convolutional neural network is loaded in ModelLoader and fit here.
    Later iterations of this code was ran over a 6 GPU (nvidia 1080s) cluster we built in our local office.
'''
import pdb
import keras.backend as K
import numpy as np
from EventFiller import EventFiller
from ModelLoader import ModelLoader
from Utilities import Utilities
import sklearn.metrics as skmetrics
from multiprocessing.pool import ThreadPool
import pandas as pd
import energyflow as ef
from energyflow.archs import PFN
from energyflow.datasets import qg_jets
from energyflow.utils import data_split, remap_pids, to_categorical
from glob import glob
from History.utils import Histories
from keras.utils import plot_model
import ConfigParser
import argparse
from Trainer import store_results

parser = argparse.ArgumentParser()
parser.add_argument('--config', default='config_deepsets.ini',
                        help="Configuration file")
args = parser.parse_args()
configuration_name = args.config

######   Parse config:    #####
config = ConfigParser.RawConfigParser()
config.read(configuration_name)

TRAIN_DATA = config.get("data","train")
TEST_DATA = config.get("data","test")
TRAINING_RES =  config.get("model","dir")
MODEL_NAME = config.get("model","name")

def model_build(nParticles=60,nFeatures=47, Phi_sizes=(50, 50, 12), F_sizes=(50, 50, 50)):
    """


    :return:
    """
    model = PFN(input_dim=nFeatures, Phi_sizes=Phi_sizes, F_sizes=F_sizes, output_dim=1, output_act='sigmoid', loss='binary_crossentropy')
    return model


if __name__ == "__main__":
    # Specify number of particles to use and number of features
    nParticles=60
     # nFeatures=51
    nFeatures=47

    Phi_sizes, F_sizes = (50, 50, 12), (50, 50, 50)
    model = model_build(nParticles,nFeatures,Phi_sizes,F_sizes)
    plot_model(model, to_file='deepset.png')

    utils  = Utilities(nParticles)
    print("TRAIN_DATA: ", TRAIN_DATA)

    # Build the first training dataset
    X_train,Y, W_train, MVA_train = utils.BuildBatch(indir=TRAIN_DATA)
    history = Histories()
    history.set_up_config(config=config)
    history.on_train_begin()

    for epoch in range(1000):
    # Shuffle loaded datasets and begin
        inds = range(len(X_train))
        np.random.shuffle(inds)
        X_epoch,Y_epoch,W_epoch,MVA_epoch = X_train[inds],Y[inds],W_train[inds], MVA_train[inds]
        if (np.min(W_train) == np.nan) : continue
        if (np.min(W_train) == np.inf) : continue

        ##Save the validation:
        history.set_mode(mode="train")
        model.fit(X_epoch, Y_epoch, batch_size=4 * 1024, epochs=1, verbose=1, sample_weight=W_epoch)

        ##Save shape and Datasets to results
        # pd.DataFrame(X_epoch).to_csv("X_example.csv", index=False)
        # pd.DataFrame(Y_epoch).to_csv("Y_example.csv", index=False)
        # pd.DataFrame(MVA_epoch).to_csv("MVA_example.csv", index=False)

        # Write out performance on validation set
        train_pred = model.predict(X_epoch)
        store_results(model, epoch=epoch, config=config)
        history.on_epoch_end(Y_epoch, [i[0].round() for i in train_pred])
        if (epoch % 10 == 0):
            history.store_history()
            # X_train,Y,W_train, MVA = utils.BuildBatch(indir=TRAIN_DATA)
            df_preds = pd.DataFrame({"train_pred": [i[0] for i in train_pred]})
            df_preds.to_csv("{1}/train_prediction_e_{0}.csv".format(epoch, TRAINING_RES))
            df_label = pd.DataFrame({'labels_train_e_{0}'.format(epoch): [i for i in Y_epoch]})
            df_label.to_csv("{1}/labels_train_e_{0}.csv".format(epoch, TRAINING_RES), index=False)
            df_mva = pd.DataFrame({'mva_train_e_{0}'.format(epoch): [i for i in MVA_epoch]})
            df_mva.to_csv("{1}/labels_mva_e_{0}.csv".format(epoch, TRAINING_RES), index=False)
        X_train, Y, W_train, MVA = utils.BuildBatch(indir=TRAIN_DATA)


