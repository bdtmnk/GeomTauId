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
from keras.utils import plot_model

MODEL_DIR = "./DeepSets/"

if __name__ == "__main__":
    # Specify number of particles to use and number of features
    nParticles=60
     # nFeatures=51
    nFeatures=47

    Phi_sizes, F_sizes = (50, 50, 12), (50, 50, 50)
    model = PFN(input_dim=nFeatures, Phi_sizes=Phi_sizes, F_sizes=F_sizes, output_dim=1, output_act='sigmoid', loss='binary_crossentropy')
    plot_model(model, to_file='deepset.png')

    utils  = Utilities(nParticles)

    # Build the first training dataset
    X_train,Y, W_train, MVA_train = utils.BuildBatch()
    print(MVA_train.shape)

    for epoch in range(10000):
    # Shuffle loaded datasets and begin
        inds = range(len(X_train))
        np.random.shuffle(inds)
        X_epoch,Y_epoch,W_epoch,MVA_epoch = X_train[inds],Y[inds],W_train[inds], MVA_epoch[inds]
        if (np.min(W_train) == np.nan) : continue
        if (np.min(W_train) == np.inf) : continue

        model.fit(X_epoch, Y_epoch,
            epochs=1,
            batch_size=4*512,
            verbose=1)
    	pd.DataFrame(X_epoch).to_csv("X_example.csv", index=False)
    	pd.DataFrame(Y_epoch).to_csv("Y_example.csv", index=False)
    	pd.DataFrame(MVA_epoch).to_csv("MVA_example.csv", index=False)
        if (epoch%10 ==0) :

            model.save('/beegfs/desy/user/dydukhle/TauId/models/keras__deep_set__%i.model' % (epoch))
            preds = model.predict(X_epoch,batch_size=1*2048)
            mva =  MVA_train
            df_preds = pd.DataFrame({"pred":[i[0] for i in preds], "labels":[y for y in Y_epoch]})#, 'mva':[m[0] for m in mva] })
            df_preds.to_csv("./DeepSets_res/labels_and_pred_e_{0}.csv".format(epoch))
            df_mva = pd.DataFrame(mva)
            df_mva.to_csv("./DeepSets_res/mva_{0}.csv".format(epoch))
    # Get next batch from background loader res
        X_train,Y,W_train, MVA_train = utils.BuildBatch()#res.get(timeout=180)

