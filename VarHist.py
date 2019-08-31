"""
Writen by Leonid Didukh
Modified by K. Bukyn
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import uproot
import pandas as pd
import os
from glob import glob
import random

plt.ioff()
matplotlib.use('Agg')

var = "nPFCands_1"

el_match = "lepEleMatch_1"
mu_match = "lepMuMatch_1"
tau_match = "lepTauMatch_1"
jet_match = "jetMatch_1"
dms = "decayMode_1"

proces = ['WJ', 'QCD', 'DYI', 'DYJ', 'Z', 'Rare']
classes = ['el', 'mu', 'tau', 'had']# 'had_5', 'had_6', 'had_0', 'had_1', 'had_10']
class_hist = {'el':[],
              'mu':[],
              'tau':[],
              'jet':[]}

TRAIN_SET = "/nfs/dust/cms/user/dydukhle/TauIdSamples/TauId/2016/train_samples/"
TRAINING_RES = "/nfs/dust/cms/user/bukinkir/TauId/histograms/"

FEATURES = [
         'nLooseTaus',
         'nPFCands_1',
         'decayMode_1',
         'lepRecoPt_1',
         'lepRecoEta_1',
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
         'pfCandPtRelPtRel_1',
         'pfCandD0D0_1',
         'pfCandDzDz_1',
         'pfCandD0Dz_1',
         'pfCandD0Dphi_1',
         'pfCandPuppiWeight_1',
         'pfCandHits_1',
         'pfCandPixHits_1',
         'pfCandLostInnerHits_1',
         'pfCandDVx_1',
         'pfCandDVy_1',
         'pfCandDVz_1',
         'pfCandD_1',
         'pfCandPdgid_1',
         'pfCandCharge_1',
         'pfCandFromPV_1',
         'pfCandVtxQuality_1',
         'pfCandTauIndMatch_1',
         'pfCandHighPurityTrk_1',
         'pfCandIsBarrel_1',
         'lepHasSV_1'
    ]

VECT_FEATURES = [
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
         'pfCandPtRelPtRel_1',
         'pfCandD0D0_1',
         'pfCandDzDz_1',
         'pfCandD0Dz_1',
         'pfCandD0Dphi_1',
         'pfCandPuppiWeight_1',
         'pfCandHits_1',
         'pfCandPixHits_1',
         'pfCandLostInnerHits_1',
         'pfCandDVx_1',
         'pfCandDVy_1',
         'pfCandDVz_1',
         'pfCandD_1',
         'pfCandPdgid_1',
         'pfCandCharge_1',
         'pfCandFromPV_1',
         'pfCandVtxQuality_1',
         'pfCandTauIndMatch_1',
         'pfCandHighPurityTrk_1',
         'pfCandIsBarrel_1',
    ]

##WJ file list with the jets:
WJ_list = glob(TRAIN_SET +"WJ*.root")
## DY file list
DY_list = glob(TRAIN_SET+"DY*.root")


def read_file(file_names):
    # random.shuffle(file_names)
    # file_names = file_names[]
    dm = "decayMode_1"
    el_match = "lepEleMatch_1"
    mu_match = "lepMuMatch_1"
    tau_match = "lepTauMatch_1"
    for file_name in file_names:
        root_file = uproot.open(file_name)['Candidates']
        df = root_file.pandas.df(entrystop=1000, flatten=False)   # not the default
        df[VECT_FEATURES] = df[VECT_FEATURES].applymap(lambda x: x[0])
        for var in VECT_FEATURES:
            if df[var].dtype == 'object':
                df[var] = df[var].astype('float32')

        # print(np.argmax(df['pfCandPt_1'].iloc[0]))
        if "WJ" in file_name:
            df = df[(df[mu_match]==0)&(df[el_match]==0)&(df[tau_match]==0)&((df[dm] <= 1) | (df[dm] == 10))]
        elif "DY" in file_name:
            df = df[(df[mu_match]==0)&(df[el_match]==0)&(df[tau_match]==1)&((df[dm] <= 1) | (df[dm] == 10))]
        yield df


def plot_dist(df, var):
    fig, (ax, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 4))

    ax.hist(df[(df['decay_mode'] == 0)][(df['label'] == 1)][var], bins=15, log=True, color='r',
            label='MVA True')
    ax.hist(df[(df['decay_mode'] == 0)][(df['label'] == 0)][var], bins=15, log=True, color='b',
            label='MVA Fakes')
    plt.title("Decay Mode: {0}".format(0))
    ax.legend()
    plt.xlabel(var)
    plt.ylabel("Arn Units")

    ax2.hist(df[(df['decay_mode'] == 1)][(df['label'] == 1)][var], bins=30, log=True, color='r',
             label="PF True")
    ax2.hist(df[(df['decay_mode'] == 1)][(df['label'] == 0)][var], bins=30, log=True, color='b',
             label="PF Fakes")
    plt.title("Decay Mode: {0}".format(1))

    ax3.hist(df[(df['decay_mode'] == 10)][(df['label'] == 1)][var], bins=30, log=True, color='r',
             label="PF True")
    ax3.hist(df[(df['decay_mode'] == 10)][(df['label'] == 0)][var], bins=30, log=True, color='b',
             label="PF Fakes")
    plt.title("Decay Mode: {0}".format(10))

    plt.legend()
    plt.xlabel("Output")
    plt.ylabel("Arn Units")
    plt.savefig('{0}Hist_{1}_dm.pdf'.format(TRAINING_RES, var))

    plt.figure()
    plt.hist(df[(df['decay_mode'] == 0)][(df['label'] == 1)][var], bins=15, log=True, color='r',
            label='MVA True')
    plt.hist(df[(df['decay_mode'] == 0)][(df['label'] == 0)][var], bins=15, log=True, color='b',
            label='MVA Fakes')
    plt.xlabel(var)
    plt.ylabel("Arn Units")
    plt.legend()
    # plt.title('Inverted ROC')
    plt.savefig('{0}Hist_{1}.pdf'.format(TRAINING_RES, var))
    return

def plot_scalar_variables(df):
    """
    """

    return None


def plot_vector_variable(df, var_name):
    """
    Sort Vectors by PT:
    """
    var_df = df[var_name]
    ##Extract index:
    # max_val = sort(var_df)[-1]

    return None

if __name__=='__main__':
    file_list = WJ_list + DY_list
    df_gen = read_file(file_list)
    df = pd.DataFrame()
    for _df in df_gen:
        df = df.append(_df, ignore_index=True)
    df.reindex()
    df.to_csv(
        "{0}data.csv".format(TRAINING_RES),
        index=False)
    # for feature in FEATURES:
    #     plot_dist(df, feature)
