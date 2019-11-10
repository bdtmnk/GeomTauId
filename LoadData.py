import random
from glob import glob

import numpy as np
import pandas as pd
import torch
import uproot
from pandas.api.types import CategoricalDtype
from torch_geometric.data import Dataset, Data

TRAINING_RES = "/nfs/dust/cms/user/bukinkir/TauId/histograms/"

COORDINATES = np.char.array([
    'pfCandDEta_1',
    'pfCandDPhi_1',
    'pfCandEta_1'
])

FEATURES = np.char.array([
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
         'pfCandPuppiWeight_1',
         'pfCandHits_1',
         'pfCandPixHits_1',
         'pfCandDVx_1',
         'pfCandDVy_1',
         'pfCandDVz_1',
         'pfCandD_1'
    ])

BINARY_FEATURES = np.char.array([
    'pfCandTauIndMatch_1',
    'pfCandHighPurityTrk_1',
    'pfCandIsBarrel_1',
    'lepHasSV_1'
])

CATEGORICAL_FEATURES = np.char.array([
    'decayMode_1',
    'pfCandCharge_1',
    'pfCandLostInnerHits_1',
    'pfCandPdgid_1',
    'pfCandVtxQuality_1',
    'pfCandFromPV_1'
])

VECT_FEATURES = np.char.array([
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
    ])

TARGET = np.char.array(['lepTauMatch_1', ])



def get_indices(root_file, file_name):
    """
    Get the indices of all the appropriate events from the given file
    Only selecting tau from Drell-Yan events and jets from W+jets events, select decay modes 0, 1 and 10 for both tau and jets
    :param root_file: Opened ROOT file
    :param file_name: Name of the ROOT file
    :return: List of the indices
    """

    df = pd.DataFrame()
    df['decay_mode'] = root_file['decayMode_1'].array()
    df['el_match'] = root_file["lepEleMatch_1"].array()
    df['mu_match'] = root_file["lepMuMatch_1"].array()
    df['tau_match'] = root_file["lepTauMatch_1"].array()
    df['jet_match'] = 1 - df['el_match'] - df['tau_match'] - df['mu_match']
    index = 0
    if "DY" in file_name:
        index = np.where((df['tau_match'] == 1))[0]# & ((df['decay_mode'] <= 1) | (df['decay_mode'] == 10)))[0]
    else:# "WJ" in file_name:
        index = np.where((df['jet_match'] == 1))[0]# & ((df['decay_mode'] <= 1) | (df['decay_mode'] == 10)))[0]
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


class TauIdDataset(Dataset):
    """Class for data loading from disk."""
    
    def __init__(self, root, mode='train', num = 1024, nfiles = 1, processes="all", scale=False):
        """
        :param root: Path to directory where ROOT files with data are stored
        :param mode: 'train' or 'test', in second case loads all the variables from tree with labels for evaluation
        :param num: Number of events to be used in one epoch
        :param nfiles: Number of files to load (the same number is used for signal and background files)
        """
        self.signal = glob(root+"DY*.root")
        
        if self.processes == "all":
            self.bkg_files = glob(root+"WJ*.root") + glob(root+"QCD*.root") + glob(root+"Top*.root")
        
        elif self.processes == "WJ":
            self.bkg_files = glob(root+"WJ*.root")
        
        
        self.filenames = []
        
        for i in range(nfiles):
            filename = random.choice(self.sig_files)
            self.sig_files.remove(filename)
            self.filenames.append(filename)
        
        for i in range(nfiles):
            filename = random.choice(self.bkg_files)
            self.bkg_files.remove(filename)
            self.filenames.append(filename)

        self.root = root
        self.len = num
        self.mode = mode
        if self.mode == 'test':self.test_data = []

        if scale:
            self.max = pd.read_csv("{0}max.csv".format(TRAINING_RES))['val'].astype('float32')
            self.features = pd.read_csv("{0}max.csv".format(TRAINING_RES))['feature']
            self.min = pd.read_csv("{0}min.csv".format(TRAINING_RES))['val'].astype('float32')
        self.nfiles = nfiles*2
        
        self.cat_types = pd.Series([ CategoricalDtype(categories=[0, 1, 10], ordered = True),
                                     CategoricalDtype(categories=[-1, 0, 1], ordered=True),
                                     CategoricalDtype(categories=[-1, 0, 1, 2], ordered=True),
                                     CategoricalDtype(categories=[1, 2, 11, 13, 130, 211, 22], ordered=True),
                                     CategoricalDtype(categories=[1, 5, 6, 7], ordered=True),
                                     CategoricalDtype(categories=[1, 2, 3], ordered=True)], index=CATEGORICAL_FEATURES)
        # self.files = []
        self.indices = []
        self.data = []
        for i in range(self.nfiles):
            # self.files.append(uproot.open(self.filenames[i]))
            # self.indices.append(get_indices(self.files[i]['Candidates'], self.filenames[i]))
            # self.data.append(self.files[i]['Candidates'].pandas.df(
            #     np.concatenate((FEATURES, BINARY_FEATURES, CATEGORICAL_FEATURES, TARGET))).loc[self.indices[i]].astype(
            #     'float32'))
            # self.data[i] = self.pre_process(self.data[i])

            file = uproot.open(self.filenames[i])
            self.indices.append(get_indices(file['Candidates'], self.filenames[i]))
            if self.mode == 'train':
                self.data.append(file['Candidates'].pandas.df(
                    np.concatenate((FEATURES, BINARY_FEATURES, CATEGORICAL_FEATURES, TARGET))).loc[self.indices[i]].astype(
                    'float32'))
                self.data[i] = self.pre_process(self.data[i])
            elif self.mode == 'test':
                self.data.append(file['Candidates'].pandas.df(np.concatenate((FEATURES, BINARY_FEATURES, CATEGORICAL_FEATURES, TARGET, ['lepMVAIso_1',]))).loc[self.indices[i]].astype('float32'))
                self.test_data.append(self.data[i])
                self.data[i] = self.pre_process(self.data[i])
            # print(self.data[i].memory_usage(index=True).sum())
        # print(self.data)
        # print(self.indices)

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
        # for i in range(self.nfiles):
        #     print(len(self.indices[i]))
        i = random.randint(0, self.nfiles - 1)
        # root_file = uproot.open(self.filenames[index])
        # root_file = self.files[i]
        if (len(self.indices[i]) % 100) == 0:
            print(len(self.indices[i]))
        if len(self.indices[i]) > 0:
            j = random.choice(self.indices[i])
            # print(np.where(self.indices[i] == j))
            self.indices[i] = np.delete(self.indices[i], np.where(self.indices[i] == j)[0])
            if self.mode == 'train':
                return self.get_tensor(self.data[i].loc[j])
            elif self.mode == 'test':
                return self.get_tensor(self.data[i].loc[j], self.test_data[i].loc[j])
        else:
            self.reload_file(i)
            print(self.filenames)
            print(self.filenames)
            i = self.nfiles - 1
            j = random.choice(self.indices[i])
            # print(np.where(self.indices[i] == j))
            self.indices[i] = np.delete(self.indices[i], np.where(self.indices[i] == j)[0])
            if self.mode == 'train':
                return self.get_tensor(self.data[i].loc[j])
            elif self.mode == 'test':
                return self.get_tensor(self.data[i].loc[j], self.test_data[i].loc[j])

    def __len__(self):
        return self.len

    def reload_file(self, index):
        """
        Load new file instead of file with given index.
        Currently function isn't working properly
        :param index: Index of file to delete.
        :return: None
        """
        filename = self.filenames[index]
        self.indices.remove(self.indices[index])
        self.data.remove(self.data[index])
        self.filenames.remove(self.filenames[index])
        if "DY" in filename:
            filename = random.choice(self.sig_files)
            self.sig_files.remove(filename)
            self.filenames.append(filename)
        elif "WJ" in filename:
            filename = random.choice(self.bkg_files)
            self.bkg_files.remove(filename)
            self.filenames.append(filename)
        file = uproot.open(filename)
        i = self.nfiles - 1
        self.indices.append(get_indices(file['Candidates'], filename))
        if self.mode == 'train':
            self.data.append(file['Candidates'].pandas.df(
                np.concatenate((FEATURES, BINARY_FEATURES, CATEGORICAL_FEATURES, TARGET))).loc[self.indices[i]].astype(
                'float32'))
            self.data[i] = self.pre_process(self.data[i])
        elif self.mode == 'test':
            self.data.append(file['Candidates'].pandas.df().loc[self.indices[i]].astype('float32'))
            self.test_data.append(self.data[i])
            self.data[i] = self.pre_process(self.data[i])

    def norm_min_max(self, value):
        """
        Normalise given feature.
        :param value: Pandas series with feature to normalise
        :return: Pandas series with normalised feature
        """
        feature = value.name
        min = self.min.get(self.features[self.features == feature].index[0])
        max = self.max.get(self.features[self.features == feature].index[0])
        return (value - min) / (max - min)

    def set_category(self, value):
        """
        Apply one-hot encoding to categorical feature.
        :param value: Pandas series with categorical feature
        :return: Dataframe with encoded feature
        """
        return  value.astype(self.cat_types[value.name])

    def get_tensor(self, df, df_test=None):
        """
        Transform dataframe to pytorch tensor with features and labels.
        :param df: Dataframe to transform
        :param df_test: Dataframe with not pre-transformed features (only needed if mode='test')
        :return: Pytorch tensor
        """
        label = df['lepTauMatch_1'].iloc[:1]
        df = df.drop(columns=TARGET)
        pos = torch.tensor(df[COORDINATES].values)
        x = torch.tensor(df.values)

        data = Data()
        data.pos = pos
        data.x = x
        
        if self.mode == 'train':
            data.y = torch.tensor(label.values,  dtype=torch.int64)
        elif self.mode == 'test':
            df_ = pd.DataFrame( [df_test['lepTauMatch_1'][0],
                                df_test['lepMVAIso_1'][0],
                                df_test['nLooseTaus'][0],
                                df_test['nPFCands_1'][0],
                                df_test['decayMode_1'][0],
                                df_test['lepRecoPt_1'][0],
                                df_test['lepRecoEta_1'][0],
                                df_test['pfCandPt_1'][0],
                                df_test['pfCandPz_1'][0],
                                df_test['pfCandPtRel_1'][0],
                                df_test['pfCandPzRel_1'][0],
                                df_test['pfCandDr_1'][0],
                                df_test['pfCandDEta_1'][0],
                                df_test['pfCandDPhi_1'][0],
                                df_test['pfCandEta_1'][0],
                                df_test['pfCandDz_1'][0],
                                df_test['pfCandDzErr_1'][0],
                                df_test['pfCandDzSig_1'][0],
                                df_test['pfCandD0_1'][0],
                                df_test['pfCandD0Err_1'][0],
                                df_test['pfCandD0Sig_1'][0],
                                df_test['pfCandPuppiWeight_1'][0],
                                df_test['pfCandHits_1'][0],
                                df_test['pfCandPixHits_1'][0],
                                df_test['pfCandLostInnerHits_1'][0],
                                df_test['pfCandDVx_1'][0],
                                df_test['pfCandDVy_1'][0],
                                df_test['pfCandDVz_1'][0],
                                df_test['pfCandD_1'][0],
                                df_test['pfCandPdgid_1'][0],
                                df_test['pfCandCharge_1'][0],
                                df_test['pfCandFromPV_1'][0],
                                df_test['pfCandVtxQuality_1'][0],
                                df_test['pfCandTauIndMatch_1'][0],
                                df_test['pfCandHighPurityTrk_1'][0],
                                df_test['pfCandIsBarrel_1'][0],
                                df_test['lepHasSV_1'][0]
                                 ])
            data.y = torch.tensor(df_.values)
            data.y = torch.transpose(data.y, 0, 1)
        return data

    def pre_process(self, df):
        """
        Pre-process the input data.
        :param df: Dataframe with not pre-transformed features
        :return: Dataframe with pre-transformed features
        """
        if self.mode == 'test':
            df = df[np.concatenate((FEATURES, BINARY_FEATURES, CATEGORICAL_FEATURES, TARGET))]
        df[FEATURES] = df[FEATURES].apply(self.norm_min_max, axis=0)
        df = pd.concat([df[np.concatenate((FEATURES, BINARY_FEATURES, TARGET))],
                        pd.get_dummies(df[CATEGORICAL_FEATURES].apply(self.set_category, axis=0))], axis=1)
        df = df.fillna(0)
        return df


class LoadData:
    """Class for data loading (doesn't work currently)"""

    def __init__(self, root):
        """
        :param root: Path to directory where ROOT files with data are stored
        """
        self.sig_files = glob(root+"DY*.root")
        self.bkg_files = glob(root+"WJ*.root")
        self.cat_types = pd.Series([CategoricalDtype(categories=[0, 1, 10], ordered=True),
                                    CategoricalDtype(categories=[-1, 0, 1], ordered=True),
                                    CategoricalDtype(categories=[-1, 0, 1, 2], ordered=True),
                                    CategoricalDtype(categories=[1, 2, 11, 13, 130, 211, 22], ordered=True),
                                    CategoricalDtype(categories=[1, 5, 6, 7], ordered=True),
                                    CategoricalDtype(categories=[1, 2, 3], ordered=True)], index=CATEGORICAL_FEATURES)
        self.max = pd.read_csv("{0}max.csv".format(TRAINING_RES))['val'].astype('float32')
        self.features = pd.read_csv("{0}max.csv".format(TRAINING_RES))['feature']
        self.min = pd.read_csv("{0}min.csv".format(TRAINING_RES))['val'].astype('float32')
        
    def load_data(self, num):
        """
        Load specified number of events.
        :param num: Number of events to load
        :return: List of pytorch tensors
        """
        data = []
        while len(data) < (num / 2):
            print(len(data))
            filename = random.choice(self.sig_files)
            self.sig_files.remove(filename)
            file = uproot.open(filename)
            indices = get_indices(file['Candidates'], filename)
            df = file['Candidates'].pandas.df(
                np.concatenate((FEATURES, BINARY_FEATURES, CATEGORICAL_FEATURES, TARGET))).loc[indices].astype(
                'float32')
            df = self.pre_process(df)
            # data = df.apply(self.get_tensor)
            for i in np.unique(df.index.get_level_values('entry').to_numpy()):
                data.append(self.get_tensor(df.loc[i]))
                if len(data) >= (num / 2):
                    break
        data = data[:num/2]

        while len(data) < num:
            filename = random.choice(self.bkg_files)
            self.bkg_files.remove(filename)
            file = uproot.open(filename)
            indices = get_indices(file['Candidates'], filename)
            df = file['Candidates'].pandas.df(
                np.concatenate((FEATURES, BINARY_FEATURES, CATEGORICAL_FEATURES, TARGET))).loc[indices].astype(
                'float32')
            df = self.pre_process(df)
            # data = df.apply(self.get_tensor)
            for i in np.unique(df.index.get_level_values('entry').to_numpy()):
                data.append(self.get_tensor(df.loc[i]))
                if len(data) >= num:
                    break
        data = data[:num]
        print(data)
        return random.shuffle(data)

    def pre_process(self, df):
        """
        Pre-process the input data.
        :param df: Dataframe with not pre-transformed features
        :return: Dataframe with pre-transformed features
        """
        df[FEATURES] = df[FEATURES].apply(self.norm_min_max, axis=0)
        df = pd.concat([df[np.concatenate((FEATURES, BINARY_FEATURES, TARGET))],
                        pd.get_dummies(df[CATEGORICAL_FEATURES].apply(self.set_category, axis=0))], axis=1)
        df = df.fillna(0)
        return df
    
    def norm_min_max(self, value):
        """
        Normalise given feature.
        :param value: Pandas series with feature to normalise
        :return: Pandas series with normalised feature
        """
        feature = value.name
        min = self.min.get(self.features[self.features == feature].index[0])
        max = self.max.get(self.features[self.features == feature].index[0])
        return (value - min) / (max - min)

    def set_category(self, value):
        """
        Apply one-hot encoding to categorical feature.
        :param value: Pandas series with categorical feature
        :return: Dataframe with encoded feature
        """
        return  value.astype(self.cat_types[value.name])

    def get_tensor(self, df):
        """
        Transform dataframe to pytorch tensor with features and labels.
        :param df: Dataframe to transform
        :return: Pytorch tensor
        """
        label = df['lepTauMatch_1'].iloc[:1]
        df = df.drop(columns=TARGET)
        pos = torch.tensor(df[COORDINATES].values)
        x = torch.tensor(df.values)
        data = Data()
        data.pos = pos
        data.x = x

        data.y = torch.tensor(label.values, dtype=torch.int64)
        return data

