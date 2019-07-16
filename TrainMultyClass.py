#!/usr/bin/python
# Written by Owen Colegrove
# Edited by Leonid Didukh

import keras.backend as K
import numpy as np
from EventFiller import EventFiller
from ModelLoader import ModelLoader
from Utilities import Utilities
import sklearn.metrics as skmetrics
from multiprocessing.pool import ThreadPool
import pandas as pd
from glob import glob
from History.utils import Histories
import ConfigParser
from sklearn.utils import class_weight
def validate(files, model, nParticles):
  """

  :param files:
  :param model:
  :return:
  """
  utils = Utilities(nParticles)
  for file in files:
    proc_name = file.split("/")[-1]
    X_1, Y_1, _, MVA  = utils.BuildValidationDataset(file)
    pred = model.predict(X_1)
    pred.to_csv("{0}/{1}.csv".format(TRAINING_RES,proc_name),index=False)
    #Evaluate Results:

  return

def store_results(model, epoch, config):
  """
  Store results of training
  :return:
  """
  # if we want, adjust the learning rate as it goes ... K.set_value(model.optimizer.lr,.0001)
  # Save checkpoint of the model
  model.save(config.get("model","dir") + "/" + config.get("model","name") + "{0}.json".format(epoch))
  # serialize model to JSON
  model_json = model.to_json()
  with open(config.get("model","dir") + "/" + config.get("model","name") + "A_{0}.json".format(epoch), "w") as json_file:
    json_file.write(model_json)
  # serialize weights to HDF5
  model.save_weights(config.get("model","dir") + "/" + config.get("model","name") + "W_{0}.h5".format(epoch))
  print("Saved model to disk")
  return


if __name__ == "__main__":
  import argparse

  parser = argparse.ArgumentParser()
  parser.add_argument('--config', default='config.ini',
                      help="Configuration file")
  args = parser.parse_args()
  configuration_name = args.config

  ######   Parse config:    #####
  config = ConfigParser.RawConfigParser()
  config.read(configuration_name)

  TRAIN_DATA = config.get("data", "train")
  TEST_DATA = config.get("data", "test")
  TRAINING_RES = config.get("model", "dir")
  MODEL_NAME = config.get("model", "name")

  # Specify number of particles to use and number of features
  nParticles=60
  #nFeatures=51
  nFeatures=47
  loader = ModelLoader((nParticles,nFeatures))
  ## Define Loss for the model:
  from Loss.Loss import multi_weighted_logloss

  utils  = Utilities(nParticles)
  #history = Histories()
  #history.set_up_config(config=config)
  #history.on_train_begin()
  # Build the first training dataset
  print("TRAIN_DATA: ", TRAIN_DATA)
  X_train, Y, W_train, MVA_train = utils.BuildBatch(indir=TRAIN_DATA, nEvents=50, nFiles=10)

  model  = loader.load_multiclass(ouput_class=4, loss='categorical_crossentropy')#,weights=class_weight)

  for epoch in range(1000):
    pool_local = ThreadPool(processes=1)
    # Shuffle loaded datasets and begin
    inds = range(len(X_train))
    np.random.shuffle(inds)
    X_epoch,Y_epoch,W_epoch, MVA_epoch = X_train[inds],Y[inds],W_train[inds], MVA_train[inds]
    # Check that nothing strange happened in the loaded datset
    if (np.min(W_train) == np.nan):  continue
    if (np.min(W_train) == np.inf):  continue
    cwd = {0:1,1:1,2:1,3:1}#dict()
    ##Save the validation:
    ## Get class weights:
    Y = MVA_epoch[:,2:]
    y = np.argmax(Y, axis=1)
    _class_weight = class_weight.compute_class_weight("balanced", [0,1,2,3], y)
    cwd = _class_weight
    print("Computed CW", _class_weight)
#    class_dict = Y.sum(axis=0)
#    print(class_dict)
#    class_weight = class_dict.astype(np.float)/Y.sum(axis=0).sum()
#    cwd = {}
#    for w in range(4):cwd[w] = class_weight[w]
#    print(cwd)
    model.fit(X_epoch, Y,batch_size=4*1024, epochs=1, verbose=1,sample_weight = W_epoch, class_weight=cwd)

    ##Save shape and Datasets to results
    #pd.DataFrame(X_epoch).to_csv("X_example.csv", index=False)
    #pd.DataFrame(Y_epoch).to_csv("Y_example.csv", index=False)
    #pd.DataFrame(MVA_epoch).to_csv("MVA_example.csv", index=False)

    # Write out performance on validation set
    train_pred = model.predict(X_epoch)
    store_results(model, epoch=epoch, config=config)
    #history.on_epoch_end(Y_epoch, [i[0].round() for i in train_pred])
    if (epoch%10==0):
#      history.store_history()
      #X_train,Y,W_train, MVA = utils.BuildBatch(indir=TRAIN_DATA)
      df_preds = pd.DataFrame({"train_pred":[i for i in train_pred]})
      df_preds.to_csv("{1}/train_prediction_e_{0}.csv".format(epoch, TRAINING_RES))
      df_label = pd.DataFrame({'labels_train_e_{0}'.format(epoch):[i for i in Y_epoch]})
      df_label.to_csv("{1}/labels_train_e_{0}.csv".format(epoch, TRAINING_RES), index=False)
      df_mva = pd.DataFrame({'mva_train_e_{0}'.format(epoch):[i for i in MVA_epoch]})
      df_mva.to_csv("{1}/labels_mva_e_{0}.csv".format(epoch, TRAINING_RES), index=False)
    
    X_train,Y,W_train, MVA_train = utils.BuildBatch(indir=TRAIN_DATA, nEvents=(epoch+1)*500, nFiles=5)

