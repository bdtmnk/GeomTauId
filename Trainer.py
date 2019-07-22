#!/usr/bin/python
# Written by Owen Colegrove
'''
    The code below are scraps of actual code I used to train a deep neural network for object recognition using the CMS detector.  It greatly outperformed previous high-level techniques developed over many generations by high energy physicists.
    Raw detector information from is condensed down into a ~1 tb sample of 100M simulated particle decays with real and fake events.
    The events were generated and analyzed by us using a large cluster -- they are then stored into so called ROOT files we transfer them onto our private cluster.
    The dataset, loaded with EventFiller and Utilities.. ultimately things were modified s.t. training and event loading was done in parallel
    A deep convolutional neural network is loaded in ModelLoader and fit here.
    Later iterations of this code was ran over a 6 GPU (nvidia 1080s) cluster we built in our local office.
'''
# import keras.backend as K
import numpy as np
# from EventFiller import EventFiller
from ModelLoader import ModelLoader
from EvalModel import load_model
from Utilities import Utilities
# import sklearn.metrics as skmetrics
from multiprocessing.pool import ThreadPool
import pandas as pd
# from glob import glob
from History.utils import Histories
import ConfigParser
import argparse
import time

parser = argparse.ArgumentParser()
parser.add_argument('--config', default='config.ini', help="Configuration file")
args = parser.parse_args()
configuration_name = args.config

######   Parse config:    #####
config = ConfigParser.RawConfigParser()
config.read(configuration_name)

TRAIN_DATA = config.get("data", "train")
TEST_DATA = config.get("data", "test")
TRAINING_RES = config.get("model", "dir")
MODEL_NAME = config.get("model", "name")
epoch = config.get("model","epoch")
CONTINUE = config.get("model", "continue")

def validate(files, model, nParticles):
    """

    :param files:
    :param model:
    :return:
    """
    utils = Utilities(nParticles)
    for file in files:
        proc_name = file.split("/")[-1]
        X_1, Y_1, _, MVA = utils.BuildValidationDataset(file)
        pred = model.predict(X_1)
        pred.to_csv("{0}/{1}.csv".format(TRAINING_RES, proc_name), index=False)
        # Evaluate Results:

    return


def store_results(model, epoch, config):
    """
    Store results of training
    :return:
    """
    # if we want, adjust the learning rate as it goes ... K.set_value(model.optimizer.lr,.0001)
    # Save checkpoint of the model
    model.save(config.get("model", "dir") + "/" + config.get("model", "name") + "{0}.json".format(epoch))
    # serialize model to JSON
    model_json = model.to_json()
    with open(config.get("model", "dir") + "/" + config.get("model", "name") + "A_{0}.json".format(epoch),
              "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(config.get("model", "dir") + "/" + config.get("model", "name") + "W_{0}.h5".format(epoch))
    print("Saved model to disk")
    return


if __name__ == "__main__":
    # Specify number of particles to use and number of features
    nParticles = 60
    # nFeatures=51
    nFeatures = 47
    if CONTINUE == "yes":
        model = load_model(config=config, epoch=epoch)
        print("Model Loaded")
    else:
        loader = ModelLoader((nParticles, nFeatures))
        model = loader.load()
    utils = Utilities(nParticles)
    history = Histories()
    history.set_up_config(config=config)
    history.on_train_begin()
    # Build the first training dataset
    print("TRAIN_DATA: ", TRAIN_DATA)
    X_train, Y, W_train, MVA_train = utils.BuildBatch(indir=TRAIN_DATA, nEvents=100)

    log_name = TRAINING_RES + "train_" + MODEL_NAME + ".log"
    log = open(log_name, 'w')
    start_time = time.clock()

    for epoch in range(int(epoch) + 1, 50):
        pool_local = ThreadPool(processes=1)
        # Shuffle loaded datasets and begin
        inds = range(len(X_train))
        np.random.shuffle(inds)
        X_epoch, Y_epoch, W_epoch, MVA_epoch = X_train[inds], Y[inds], W_train[inds], MVA_train[inds]

        # Check that nothing strange happened in the loaded datset
        if (np.min(W_train) == np.nan):  continue
        if (np.min(W_train) == np.inf):  continue

        ##Save the validation:
        history.set_mode(mode="train")

        model.fit(X_epoch, Y_epoch, batch_size=1000, epochs=1, verbose=1, sample_weight=W_epoch)

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
        X_train, Y, W_train, MVA = utils.BuildBatch(indir=TRAIN_DATA, nEvents=100)

        epoch_time = time.clock() - start_time
        log.write(
          "Epoch: {0}; Time: {1} s; Loss: {2}; Acc: {3}; AUC: {4}; Precision: {5}; Recall: {6}\n".format(epoch, epoch_time,
                                                                                                         history.losses['train'][-1],
                                                                                                         history.acc['train'][-1],
                                                                                                         history.aucs['train'][-1],
                                                                                                         history.prec['train'][-1],
                                                                                                         history.rec['train'][-1]))
        log.flush()

    log.close()
