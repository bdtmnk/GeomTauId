import argparse
parser = argparse.ArgumentParser()

parser.add_argument("--config", default="config.ini", help="Configuration file")
parser.add_argument("--file", default="DYInc_2016_25_tree.root", help="Root file to Evaluate")
parser.add_argument("--epoch", default="-1", help="Number of epoch")

args = parser.parse_args()
configuration_name = args.config
epoch = args.epoch
print(configuration_name)
file_to_evaluate = args.file

from Utilities import Utilities
import pandas as pd
from glob import glob
import ConfigParser
import os
from keras.models import model_from_json
from keras.optimizers import Adam
from ModelLoader import auc

#####   Parse config:    #####
config = ConfigParser.RawConfigParser()
config.read(configuration_name)
print(config)

TEST_DATA = config.get("data","test")
TRAINING_RES =  config.get("model","dir")+config.get("model","name")
MODEL_NAME = config.get("model","name")
if epoch == "-1":
    epoch = config.get("model", "epoch")


def load_model(config, epoch=0, model=None):

    dir = config.get("model", "dir")
    model_name = config.get("model", "name")
    if model is None:
        json_file = open(os.path.join(dir + "/" + model_name + 'A_{0}.json'.format(epoch)), 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
        print("Builded from json")
    print(dir + "/" + model_name + "W_{0}.h5".format(epoch))
    model.load_weights(dir + "/" + model_name + "W_{0}.h5".format(epoch))
    print("Loaded model from disk")
    opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss='binary_crossentropy', metrics=['accuracy', auc], optimizer=opt)
    return model


if __name__ == "__main__":
    if not os.path.exists("{0}/ResultsClassFilter/".format(config.get("model", "dir"))):
        os.mkdir("{0}/ResultsClassFilter/".format(config.get("model", "dir")))
    # Specify number of particles to use and number of features
    nParticles = 60
    print("Start")
    utils  = Utilities(nParticles)
    TEST_DATA = config.get("data","test") + "/" + file_to_evaluate
    print(TEST_DATA)
    # Build the validation dataset
    Samples = glob(TEST_DATA)
    print(Samples)
    if config.get("model","meta_name") == "DeepSets":
        # Build Architecture from the
        from Trainer_DeepSet import model_build
        model = model_build()
        model = load_model(config=config, epoch=epoch, model=model)
    else:
        model = load_model(config=config, epoch=epoch)
    print("Model Loaded")

    print("Sample", Samples)
    for sample in Samples:

        print(sample)
        X_valid,Y_valid,_, MVA = utils.BuildValidationDataset(sample, None)
        #print("MVA:", MVA.shape)
        #df_valid = pd.DataFrame({'DY_labels_valid':[i for i in Y_valid]})
        #print("Valid shape:", df_valid.shape)
        #df_valid.to_csv("{0}.csv".format(sample), index=False)
        #del df_valid
        predict = model.predict(X_valid, batch_size=1000)
        #print(predict)
        df_predict = pd.DataFrame({'valid_pred': [i[0] for i in predict ],
                                   'labels_valid': [i for i in Y_valid],
                                   'mva': [i[0] for i in MVA],
                                   'decay_mode': [i[1] for i in MVA],
                                   'mu_match': [i[2] for i in MVA],
                                   'el_match': [i[3] for i in MVA],
                                   'tau_match': [i[4] for i in MVA]
                                   })
        df_predict.to_csv("{2}/ResultsClassFilter/{1}_{0}.csv".format(epoch, sample.split("/")[-1], config.get("model","dir")), index=False)
   







