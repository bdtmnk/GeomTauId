from glob import glob

import numpy as np
import pandas as pd
import uproot

TRAIN_SET = "/nfs/dust/cms/user/dydukhle/TauIdSamples/TauId/2016/train_samples/"
TRAINING_RES = "/nfs/dust/cms/user/bukinkir/TauId/histograms/"

# WJ file list with the jets:
WJ_list = glob(TRAIN_SET +"WJ*.root")
# DY file list
DY_list = glob(TRAIN_SET+"DY*.root")

FEATURES = [
         'nLooseTaus',
         'nPFCands_1',
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
         'pfCandDVx_1',
         'pfCandDVy_1',
         'pfCandDVz_1',
         'pfCandD_1'
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
         'pfCandDVx_1',
         'pfCandDVy_1',
         'pfCandDVz_1',
         'pfCandD_1'
    ]


def read_max(file_names):
    """
    Find maximal values for all the specified features.

    :param file_names: Names of files among which to find the maximum
    :return: Maximal values for all the specified features in each file (list with length equal to the number of files)
    """
    # random.shuffle(file_names)
    # file_names = file_names[]
    dm = "decayMode_1"
    el_match = "lepEleMatch_1"
    mu_match = "lepMuMatch_1"
    tau_match = "lepTauMatch_1"
    for file_name in file_names:
        root_file = uproot.open(file_name)['Candidates']
        df = root_file.pandas.df(entrystop=1000, flatten=False)   # not the default
        if "WJ" in file_name:
            df = df[(df[mu_match]==0)&(df[el_match]==0)&(df[tau_match]==0)&((df[dm] <= 1) | (df[dm] == 10))]
        elif "DY" in file_name:
            df = df[(df[mu_match]==0)&(df[el_match]==0)&(df[tau_match]==1)&((df[dm] <= 1) | (df[dm] == 10))]
        df = df.filter(items = FEATURES)
        df[VECT_FEATURES] = df[VECT_FEATURES].applymap(lambda x: max(x))
        df = df.apply(np.max, axis=0)
        yield df


def read_min(file_names):
    """
    Find minimal values for all the specified features.

    :param file_names: Names of files among which to find the minimum
    :return: Minimal values for all the specified features in each file (list with length equal to the number of files)
    """
    # random.shuffle(file_names)
    # file_names = file_names[]
    dm = "decayMode_1"
    pt = "lepRecoPt_1"
    el_match = "lepEleMatch_1"
    mu_match = "lepMuMatch_1"
    tau_match = "lepTauMatch_1"
    for file_name in file_names:
        root_file = uproot.open(file_name)['Candidates']
        df = root_file.pandas.df(entrystop=1000, flatten=False)   # not the default
        if "WJ" in file_name:
            df = df[(df[mu_match]==0)&(df[el_match]==0)&(df[tau_match]==0)&((df[dm] <= 1) | (df[dm] == 10))]
        elif "DY" in file_name:
            df = df[(df[mu_match]==0)&(df[el_match]==0)&(df[tau_match]==1)&((df[dm] <= 1) | (df[dm] == 10))]
        df = df.filter(items=FEATURES)
        df[VECT_FEATURES] = df[VECT_FEATURES].applymap(lambda x: min(x))
        df = df.apply(np.min, axis=0)
        yield df


if __name__ == '__main__':
    file_list = WJ_list + DY_list
    df_gen = read_max(file_list)
    df = pd.DataFrame()
    for _df in df_gen:
        df = df.append(_df, ignore_index=True)
    df.reindex()
    df = df.apply(np.max, axis=0)
    df.to_csv("{0}max.csv".format(TRAINING_RES), header='val')

    df_gen = read_min(file_list)
    df = pd.DataFrame()
    for _df in df_gen:
        df = df.append(_df, ignore_index=True)
    df.reindex()
    df = df.apply(np.min, axis=0)
    df.to_csv("{0}min.csv".format(TRAINING_RES), header='val')
