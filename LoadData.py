import pandas as pd
import numpy as np
import random
from torch_geometric.data import Dataset, Data, InMemoryDataset
from glob import glob
import uproot
import torch

TRAINING_RES = "/nfs/dust/cms/user/bukinkir/TauId/histograms/"

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


def index_choice(root_file, file_name):
    """

    :param root_file:
    :param file_name:
    :return:
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
        # index = np.where(df['tau_match'] == 0)[0]
        index = random.choice(index)
    elif "DY" in file_name:
        index = np.where((df['tau_match'] == 1) & ((df['decay_mode'] <= 1) | (df['decay_mode'] == 10)))[0]
        index = random.choice(index)
    return index


class TauIdDataset(InMemoryDataset):
    
    def __init__(self, root, mode='train', num = 1024,transform=None, pre_transform=None):
        # super(TauIdDataset, self).__init__()
        filenames = glob(root+"DY*.root") +  glob(root+"WJ*.root")
        self.filenames = []
        for i in range(num):
            self.filenames.append(random.choice(filenames))
        self.root = root
        self.len = len(self.filenames)
        self.mode = mode
        self.max = pd.read_csv("{0}max.csv".format(TRAINING_RES), dtype='float32')['val']
        self.min = pd.read_csv("{0}min.csv".format(TRAINING_RES), dtype='float32')['val']
        # print(self.max)
        # print(self.min)
        # self.num = num

    @property
    def raw_file_names(self):
        return self.filenames
    
    @property
    def processed_file_names(self):
        return ['training.pt', 'test.pt']

    def __getitem__(self, index):
        """
        """
        root_file = uproot.open(self.filenames[index])
        data = self.process(root_file, self.filenames[index])
        return data

    def __len__(self):
        """
        """
        return self.len
    
    def process(self, root_file, file_name):
        """
        
        """
        # data_list = []
        entrystart =  index_choice(root_file['Candidates'], file_name)
        df = root_file['Candidates'].iterate(entrystart=entrystart, entrystop=entrystart+1)
        df = df.next()
        x_list = []
        max_event = 60
        nCand = len(df['pfCandCharge_1'][0])
        # Split the Features and Labels
        #val = np.zeros((max_event, len(FEATURES)))
        for feature_index in range(len(FEATURES)):
            feature = FEATURES[feature_index]

            # try:
            #     nCand = len(df[feature][0])
            # except Exception:
            #     nCand = 1
            try:
                # print(self.min.get(feature_index))
                arr = np.pad(df[feature], [(0, nCand - 1)], mode='mean')
                arr = (arr - self.min.get(feature_index))/(self.max.get(feature_index) - self.min.get(feature_index))
                x = torch.tensor(arr).float()
                # print(x)
            except Exception:
                # print(self.min.get(feature_index))
                arr = np.pad((df[feature][0] - self.min.get(feature_index))/(self.max.get(feature_index) - self.min.get(feature_index)), [(0, 60-nCand) ], mode='constant')
                arr = (df[feature][0] - self.min.get(feature_index))/(self.max.get(feature_index) - self.min.get(feature_index))
                x = torch.tensor(arr).float()
                # print(x)
            x_list.append(x)

        # for feature_index in range(len(FEATURES)):
        #     feature = FEATURES[feature_index]
        #
        #     # try:
        #     #     nCand = len(df[feature][0])
        #     # except Exception:
        #     #     nCand = 1
        #     try:
        #         arr = np.pad(df[feature], [(0, nCand - 1)], mode='mean')
        #         x = torch.tensor(arr).float()
        #     except Exception:
        #         # arr = np.pad(df[feature][0], [(0, 60 - nCand)], mode='constant')
        #         arr = df[feature][0]
        #         x = torch.tensor(arr).float()
        #     x_list.append(x)

        # print(x_list)

        pos = torch.stack(x_list)
        pos = torch.transpose(pos, 0, 1)
        data = Data()
        data.pos = pos

        for feature_index in range(len(BINARY_FEATURES)):
            feature = BINARY_FEATURES[feature_index]

            # try:
            #     nCand = len(df[feature][0])
            # except Exception:
            #     nCand = 1
            try:
                arr = np.pad(df[feature], [(0, nCand - 1)], mode='mean')
                x = torch.tensor(arr).float()
            except Exception:
                # arr = np.pad(df[feature][0], [(0, 60 - nCand)], mode='constant')
                arr = df[feature][0]
                x = torch.tensor(arr).float()
            x_list.append(x)

        # print(x_list)

        # arrs = []
        #
        # arr = df['decayMode_1']
        # arrs.append(arr == 0)
        # arrs.append(arr == 1)
        # arrs.append(arr == 10)
        #
        # for arr in arrs:
        #     arr = np.pad(arr, [(0, nCand -  1)], mode='mean')
        #     # print(arr)
        #     x_list.append(torch.tensor(arr).float())
        #
        # arrs = []
        #
        # arr = df['pfCandCharge_1'][0]
        # arrs.append(arr == -1)
        # arrs.append(arr ==  0)
        # arrs.append(arr ==  1)
        #
        # arr = df['pfCandLostInnerHits_1'][0]
        # arrs.append(arr == -1)
        # arrs.append(arr ==  0)
        # arrs.append(arr ==  1)
        # arrs.append(arr ==  2)
        #
        # arr = df['pfCandPdgid_1'][0]
        # arrs.append(arr ==  1)
        # arrs.append(arr ==  2)
        # arrs.append(arr == 11)
        # arrs.append(arr ==  13)
        # arrs.append(arr ==  130)
        # arrs.append(arr ==  211)
        # arrs.append(arr ==  22)
        # arrs.append(arr >  22)
        #
        # arr = df['pfCandVtxQuality_1'][0]
        # arrs.append(arr == 1)
        # arrs.append(arr == 5)
        # arrs.append(arr == 6)
        # arrs.append(arr == 7)
        #
        # arr = df['pfCandFromPV_1'][0]
        # arrs.append(arr == 1)
        # arrs.append(arr == 2)
        # arrs.append(arr == 3)
        # # print(x_list)
        #
        # for arr in arrs:
        #     # arr = np.pad(arr, [(0, 60 -  nCand)], mode='constant')
        #     # print(arr)
        #     x_list.append(torch.tensor(arr).float())

        x = torch.stack(x_list)
        x = torch.transpose(x, 0, 1)
        # data = Data()
        # data.pos = x
        data.x = x#, edge_index=edge_index)
        if self.mode == 'train':
            data.y = torch.tensor(df['lepTauMatch_1'],  dtype=torch.int64)
        elif self.mode == 'test':
            data.y = torch.tensor(df[['lepTauMatch_1', 'lepRecoPt_1', 'lepRecoEta_1', 'decayMode_1', 'lepMVAIso_1']])
        return data  
