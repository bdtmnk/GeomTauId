import pandas as pd
import numpy as np
import random
from torch_geometric.data import Dataset, Data, InMemoryDataset
from glob import glob
import uproot
import torch

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


def index_choice(root_file, file_name):
    """

    :param root_file:
    :param file_name:
    :return:
    """

    df = pd.DataFrame()
    df['el_match'] = root_file["lepEleMatch_1"].array()
    df['mu_match'] = root_file["lepMuMatch_1"].array()
    df['tau_match'] = root_file["lepTauMatch_1"].array()
    df['jet_match'] = 1 - df['el_match'] - df['tau_match'] - df['mu_match']
    index = 0
    if "WJ" in file_name:
        index = np.where(df['jet_match'] == 1)[0]
        # index = np.where(df['tau_match'] == 0)[0]
        index = random.choice(index)
    elif "DY" in file_name:
        index = np.where(df['tau_match'] == 1)[0]
        index = random.choice(index)
    return index


class TauIdDataset(InMemoryDataset):
    
    def __init__(self, root, mode='train',transform=None, pre_transform=None):
        # super(TauIdDataset, self).__init__()
        self.filenames = glob(root+"DY*.root") +  glob(root+"WJ*.root")
        self.root = root
        self.len = len(self.filenames)
        self.mode = mode

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
        # Split the Features and Labels
        #val = np.zeros((max_event, len(FEATURES)))
        for feature_index in range(len(FEATURES)):
            feature = FEATURES[feature_index]
            
            try:
                nCand = len(df[feature][0])
            except Exception:
                nCand = 1
            try:
                arr = np.pad(df[feature], [(0, 60-1) ], mode='constant')
                x = torch.tensor(arr).float()
            except Exception:
                arr = np.pad(df[feature][0], [(0, 60-nCand) ], mode='constant')
                x = torch.tensor(arr).float()
            x_list.append(x)
            
        x = torch.stack(x_list)
        x = torch.transpose(x, 0, 1)
        data = Data()
        data.pos = x
        data.x = x#, edge_index=edge_index)
        if self.mode == 'train':
            data.y = torch.tensor(df['lepTauMatch_1'],  dtype=torch.int64)
        elif self.mode == 'test':
            data.y = torch.tensor(pd.DataFrame(df[['lepTauMatch_1', 'lepRecoPt_1', 'lepRecoEta_1', 'decayMode_1', 'lepMVAIso_1']]))
        return data  
