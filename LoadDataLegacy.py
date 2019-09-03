import random
from glob import glob

import numpy as np
import pandas as pd
import torch
import uproot
from torch_geometric.data import Data, InMemoryDataset

TRAINING_RES = "/nfs/dust/cms/user/bukinkir/TauId/histograms/"

COORDINATES = [
    'pfCandDEta_1',
    'pfCandDPhi_1',
    'pfCandEta_1'
]

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

BINARY_FEATURES = [
    'pfCandTauIndMatch_1',
    'pfCandHighPurityTrk_1',
    'pfCandIsBarrel_1',
    'lepHasSV_1'
]

CATEGORICAL_FEATURES = [
    'decayMode_1',
    'pfCandCharge_1',
    'pfCandLostInnerHits_1',
    'pfCandPdgid_1',
    'pfCandVtxQuality_1',
    'pfCandFromPV_1'
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

VECT_FEATURES = tuple(VECT_FEATURES)
FEATURES = tuple(FEATURES)


def index_choice(root_file, file_name):
    """
    Get one random index from the given file

    Only selecting tau from Drell-Yan events and jets from W+jets events, select decay modes 0, 1 and 10 for both tau and jets
    :param root_file: Opened ROOT file
    :param file_name: Name of the ROOT file
    :return: One chosen index
    """

    df = pd.DataFrame()
    df['decay_mode'] = root_file['decayMode_1'].array()
    df['el_match'] = root_file["lepEleMatch_1"].array()
    df['mu_match'] = root_file["lepMuMatch_1"].array()
    df['tau_match'] = root_file["lepTauMatch_1"].array()
    df['jet_match'] = 1 - df['el_match'] - df['tau_match'] - df['mu_match']
    index = 0
    if "WJ" in file_name:
        index = np.where((df['jet_match'] == 1) & ((df['decay_mode'] <= 1) | (df['decay_mode'] == 10)))[0]
        index = random.choice(index)
    elif "DY" in file_name:
        index = np.where((df['tau_match'] == 1) & ((df['decay_mode'] <= 1) | (df['decay_mode'] == 10)))[0]
        index = random.choice(index)
    return index


def get_weights(pt_train, Y):
    """
    Get weights for the sample

    Reweighting is applied to flatten tau $p_T$ spectrum
    :param pt_train: $p_T$ list for the sample
    :param Y: Labels for the sample
    :return: Pytorch tensor with the weights
    """
    W_train = np.zeros(len(Y)) + 1
    bins_by_pt = np.append(np.arange(30, 100, 10), [10000])
    ptBkg = pt_train[Y != 1]
    Hbkg, xedges = np.histogram(ptBkg, bins_by_pt)
    Hbkg = 1. / Hbkg
    Hbkg[np.isnan(Hbkg)] = 1.
    Hbkg[np.isinf(Hbkg)] = 1.
    ptSig = pt_train[Y == 1]
    Hsig, xedges = np.histogram(ptSig, bins_by_pt)
    Hsig = 1. / Hsig
    Hsig[np.isnan(Hsig)] = 1.
    Hsig[np.isinf(Hsig)] = 1.
    for id_ in range(len(pt_train)):
        ptInd = np.searchsorted(xedges, pt_train[id_]) - 1
        if (Y[id_] == 1):
            W_train[id_] = Hsig[ptInd] * W_train[id_]
        else:
            W_train[id_] = Hbkg[ptInd] * W_train[id_]
    W_train[Y == 0] = W_train[Y == 0] * len(W_train) * .5 / np.sum(W_train[Y == 0])
    W_train[Y == 1] = W_train[Y == 1] * len(W_train) * .5 / np.sum(W_train[Y == 1])
    return torch.tensor(W_train, dtype=torch.float32)


class TauIdDataset(InMemoryDataset):
    """Old class for data loading from disk (work slow)."""

    def __init__(self, root, mode='train', num=1024):
        """
        :param root: Path to directory where ROOT files with data are stored
        :param mode: 'train' or 'test', in second case loads all the variables from tree with labels for evaluation
        :param num: Number of events to be used in one epoch
        """
        filenames = glob(root + "DY*.root") + glob(root + "WJ*.root")
        self.filenames = []
        for i in range(num):
            self.filenames.append(random.choice(filenames))
        self.root = root
        self.len = len(self.filenames)
        self.mode = mode
        self.max = pd.read_csv("{0}max.csv".format(TRAINING_RES))['val'].astype('float32')
        self.features = pd.read_csv("{0}max.csv".format(TRAINING_RES))['feature']
        self.min = pd.read_csv("{0}min.csv".format(TRAINING_RES))['val'].astype('float32')

    @property
    def raw_file_names(self):
        return self.filenames

    @property
    def processed_file_names(self):
        return ['training.pt', 'test.pt']

    def __getitem__(self, index):
        """
        Get event with given index.

        :param index: Index of event to select
        :return: Pytorch tensor with features and labels for one event
        """
        root_file = uproot.open(self.filenames[index])
        data = self.process(root_file, self.filenames[index])
        return data

    def __len__(self):
        return self.len

    def process(self, root_file, file_name):
        """
        Process the data

        :param root_file: Opened ROOT file
        :param file_name: Name of the ROOT file
        :return: Pytorch tensor with one event
        """
        entrystart = index_choice(root_file['Candidates'], file_name)
        df = root_file['Candidates'].iterate(entrystart=entrystart, entrystop=entrystart + 1)
        df = df.next()

        _df = pd.DataFrame()

        x_list = []
        x_pos = []
        nCand = len(df['pfCandCharge_1'][0])

        for feature_index in range(len(FEATURES)):
            feature = FEATURES[feature_index]
            min = self.min.get(self.features[self.features == feature].index[0])
            max = self.max.get(self.features[self.features == feature].index[0])

            if feature in VECT_FEATURES:
                arr = (df[feature][0] - min) / (max - min)
                x = torch.tensor(arr).float()
            else:
                arr = np.pad(df[feature], [(0, nCand - 1)], mode='mean')
                arr = (arr - min) / (max - min)
                x = torch.tensor(arr).float()
            x_list.append(x)

            if feature in COORDINATES:
                x_pos.append(x)

        pos = torch.stack(x_pos)
        pos = torch.transpose(pos, 0, 1)
        pos[torch.isnan(pos)] = 0
        data = Data()
        data.pos = pos

        for feature_index in range(len(BINARY_FEATURES)):
            feature = BINARY_FEATURES[feature_index]

            if feature in VECT_FEATURES:
                arr = df[feature][0]
                x = torch.tensor(arr).float()
            else:
                arr = np.pad(df[feature], [(0, nCand - 1)], mode='mean')
                x = torch.tensor(arr).float()
            x_list.append(x)

        arrs = []

        arr = df['decayMode_1']
        arrs.append(arr == 0)
        arrs.append(arr == 1)
        arrs.append(arr == 10)

        for arr in arrs:
            arr = np.pad(arr, [(0, nCand -  1)], mode='mean')
            x = torch.tensor(arr).float()
            x_list.append(x)

        arrs = []

        arr = df['pfCandCharge_1'][0]
        arrs.append(arr == -1)
        arrs.append(arr == 0)
        arrs.append(arr == 1)

        arr = df['pfCandLostInnerHits_1'][0]
        arrs.append(arr == -1)
        arrs.append(arr == 0)
        arrs.append(arr == 1)
        arrs.append(arr == 2)

        arr = df['pfCandPdgid_1'][0]
        arrs.append(arr == 1)
        arrs.append(arr == 2)
        arrs.append(arr == 11)
        arrs.append(arr == 13)
        arrs.append(arr == 130)
        arrs.append(arr == 211)
        arrs.append(arr == 22)
        arrs.append(arr > 22)

        arr = df['pfCandVtxQuality_1'][0]
        arrs.append(arr == 1)
        arrs.append(arr == 5)
        arrs.append(arr == 6)
        arrs.append(arr == 7)

        arr = df['pfCandFromPV_1'][0]
        arrs.append(arr == 1)
        arrs.append(arr == 2)
        arrs.append(arr == 3)

        for arr in arrs:
            x_list.append(torch.tensor(arr).float())

        x = torch.stack(x_list)
        x = torch.transpose(x, 0, 1)
        x[torch.isnan(x)] = 0
        data.x = x

        if self.mode == 'train':
            data.y = torch.tensor(df['lepTauMatch_1'], dtype=torch.int64)
        elif self.mode == 'test':
            df_ = pd.DataFrame([df['lepTauMatch_1'].astype('int32'),
                                df['lepMVAIso_1'],
                                df['nLooseTaus'],
                                df['nPFCands_1'],
                                df['decayMode_1'],
                                df['lepRecoPt_1'],
                                df['lepRecoEta_1'],
                                df['pfCandPt_1'][0],
                                df['pfCandPz_1'][0],
                                df['pfCandPtRel_1'][0],
                                df['pfCandPzRel_1'][0],
                                df['pfCandDr_1'][0],
                                df['pfCandDEta_1'][0],
                                df['pfCandDPhi_1'][0],
                                df['pfCandEta_1'][0],
                                df['pfCandDz_1'][0],
                                df['pfCandDzErr_1'][0],
                                df['pfCandDzSig_1'][0],
                                df['pfCandD0_1'][0],
                                df['pfCandD0Err_1'][0],
                                df['pfCandD0Sig_1'][0],
                                df['pfCandPtRelPtRel_1'][0],
                                df['pfCandD0D0_1'][0],
                                df['pfCandDzDz_1'][0],
                                df['pfCandD0Dz_1'][0],
                                df['pfCandD0Dphi_1'][0],
                                df['pfCandPuppiWeight_1'][0],
                                df['pfCandHits_1'][0],
                                df['pfCandPixHits_1'][0],
                                df['pfCandLostInnerHits_1'][0],
                                df['pfCandDVx_1'][0],
                                df['pfCandDVy_1'][0],
                                df['pfCandDVz_1'][0],
                                df['pfCandD_1'][0],
                                df['pfCandPdgid_1'][0],
                                df['pfCandCharge_1'][0],
                                df['pfCandFromPV_1'][0],
                                df['pfCandVtxQuality_1'][0],
                                df['pfCandTauIndMatch_1'][0].astype('int32'),
                                df['pfCandHighPurityTrk_1'][0].astype('int32'),
                                df['pfCandIsBarrel_1'][0].astype('int32'),
                                df['lepHasSV_1'].astype('int32')])
            data.y = torch.tensor([df_[0].values])
        return data