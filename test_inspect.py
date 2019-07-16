import pandas as pd
import uproot 
import os
from glob import glob
import numpy as np
import timeit

LAKE = "/beegfs/desy/user/dydukhle/TauId/new_train_samples/validation_split"
LOC = os.getcwd()
files = glob(LAKE+"/*.root")
SHAPE = {}
N_Particles = 0
N_Events = 0
start = timeit.timeit()

VARS = [
    'pfCandPt_1',
    'pfCandPz_1',
    'pfCandPtRel_1',
    'pfCandPzRel_1',
    'pfCandDr_1',
    'pfCandDEta_1',
    'pfCandDPhi_1',
    'pfCandEta_1',
    'pfCandDz_1',
    'pfCandDzErr_1',
    'pfCandDzSig_1',
    'pfCandD0_1',
    'pfCandD0Err_1',
    'pfCandD0Sig_1',
    'pfCandPixHits_1',
    'pfCandHits_1',
    'pfCandLostInnerHits_1',
    'pfCandD0D0_1',
    'pfCandDzDz_1']

TARGET=['lepMVAIso_1','lepEleMatch_1','lepMuMatch_1','lepTauMatch_1','lepTauGenMatch_1','lepDNNScore1_1',
        'pfCandTauIndMatch_1', 'decayMode_1']
WEIGHT = ['pfCandPuppiWeight_1','lepRecoPt_1']


def filler(file, FILE):
    #sortedInds = np.argsort(file['pfCandPt_1'].array())
    #batch_size = 10000#df_vars = []
    df_targets = {}
    for i in TARGET:
        df_targets[i] =  file[i].array()
    df_target = pd.DataFrame(df_targets)
    #df_targets = pd.concat(df_targets)
    print(LOC)
    print(FILE)
    df_target.to_csv("{0}/{1}.csv".format(LOC, FILE.split('/')[-1]),index=False)
    del df_targets

    return #df_vars.loc[sortedInds]


for FILE in files:
    file = uproot.open(FILE)['Candidates']
    filler(file, FILE)





