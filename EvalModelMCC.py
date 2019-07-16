import argparse
parser = argparse.ArgumentParser()

parser.add_argument("--config", default="config.ini", help="Configuration file")
parser.add_argument("--file", default="DYInc_2016_25_tree.root", help="Root file to Evaluate")

args = parser.parse_args()
configuration_name = args.config
print(configuration_name)
file_to_evaluate = args.file

from keras.models import load_model
from ModelLoader import ModelLoader
from Utilities import Utilities
import pandas as pd
from glob import glob
import numpy as np
import sklearn.metrics as skmetrics
from multiprocessing.pool import ThreadPool
from EventFiller import EventFiller
import ConfigParser
import os
from keras.models import model_from_json
from keras.optimizers import Adam

#####   Parse config:    #####
config = ConfigParser.RawConfigParser()
config.read(configuration_name)
print(config)

TEST_DATA = config.get("data","test")
TRAINING_RES =  config.get("model","dir")+config.get("model","name")
MODEL_NAME = config.get("model","name")
epoch = config.get("model","epoch")

def load_model(config, epoch=0, model=None):

    dir = config.get("model", "dir")
    model_name = config.get("model", "name")
    if model is None:
        json_file = open(os.path.join(dir + "/" + model_name + 'A_{0}.json'.format(epoch)), 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
    	print("Builded from json")
    else:
    	# load weights into new model
	print(dir + "/" + model_name + "W_{0}.h5".format(epoch))    
	model.load_weights(dir + "/" + model_name + "W_{0}.h5".format(epoch))
    print("Loaded model from disk")
    #model.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer=Adam(lr=0.01))
    return model

if __name__ == "__main__":
    # Specify number of particles to use and number of features
    if not os.path.exists("{2}/ResultsClassFilter/".format(config.get("model", "dir"))):
        os.mkdir("{2}/ResultsClassFilter/".format(config.get("model", "dir")))
    nParticles=60
    print("Start")
    utils  = Utilities(nParticles)
    TEST_DATA = config.get("data","test") + "/" +file_to_evaluate
    print(TEST_DATA)
    # Build the validation dataset
    Samples = glob(TEST_DATA)
    print(Samples)
    if config.get("model","meta_name") == "DeepSets":
        #Build Architecture from the
        from Trainer_DeepSet import model_build
        model = model_build()
        model = load_model(config=config, epoch=epoch, model=model)
    else:
        model = load_model(config=config, epoch=epoch)

    print("Model Loaded")
    print("Sample", Samples)

    for sample in Samples:
	
        print(sample)
	
        X_valid,Y_valid,_, MVA = utils.BuildValidationDataset(sample,None)
      
        #df_valid = pd.DataFrame({'DY_labels_valid':[i for i in Y_valid]})
        #print("Valid shape:", df_valid.shape)
        #df_valid.to_csv("{0}.csv".format(sample), index=False)
        #del df_valid
        predict = model.predict(X_valid, batch_size=5000)
        df_predict = pd.DataFrame({"valid_pred":[np.argmax(i) for i in predict],
                                   'mva':[i[0] for i in MVA],
                                   'decay_mode': [i[1] for i in MVA],
                                   'mu_match': [i[2] for i in MVA],
                                   'el_match': [i[3] for i in MVA],
                                   'tau_match': [i[4] for i in MVA]

                                   })
        df_predict.to_csv("{2}/ResultsClassFilter/{1}_{0}.csv".format(epoch, sample.split("/")[-1], config.get("model","dir")), index=False)

   







