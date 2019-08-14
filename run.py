#!/usr/bin/env python
# coding: utf-8
"""

Created by L.Didukh leonid.didukh@desy.de

The graph-nn model for the TauId

Input Events: n*m matrix where n - number of particle in the event, m - number of features some of them are one hot encoded.
Model: graph-nn implemented with edge-convolution operation, where nodes are connected by knn-graph.
Baseline: DPF-CNN - 1d convolution based cnn.
Metric: ROC AUC score, tau eff.

"""
import uproot 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import random
TRAIN_SET = "/nfs/dust/cms/group/susy-desy/TauId/2016/train_samples/"
TEST_SET = "/nfs/dust/cms/group/susy-desy/TauId/2016/test_samples/"

DY_train = glob(TRAIN_SET+"DY*.root")
WJ_train = glob(TRAIN_SET+"WJ*.root")
DY_test = glob(TEST_SET+"DY*.root")
WJ_test = glob(TEST_SET+"WJ*.root")

#### DataLoader: https://rusty1s.github.io/pytorch_geometric/build/html/notes/create_dataset.html#
#### Handle the DY and WJ, prompt tau from DY, fakes from WJ, Take only relevant decaying modes
#### PT cut
#### Pt Reweighting e.g. - look at Owen's implementation
#### Data Balancing? <-- Batch Configuration
#### PytTorch Data Representation - https://rusty1s.github.io/pytorch_geometric/build/html/notes/introduction.html#data-handling-of-graphs

import os.path as osp
import torch
import torch.optim as optim
from torch_geometric.data import Dataset, Data, InMemoryDataset
from torch.utils import data as D
from torch_geometric.data import Data, DataLoader
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_max_pool, global_add_pool, global_mean_pool
from torch_cluster import knn_graph
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, BatchNorm1d as BN, Dropout
from torch import nn
from torch_geometric.nn import *
from torch_geometric.nn import knn_graph
from torch_geometric.utils import to_undirected
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from torch_geometric.nn import *
from torch_geometric.utils import *
from torch_geometric.nn import GCNConv
from torch_cluster import graclus_cluster
from torch_geometric.nn import global_max_pool
### data.x: Node feature matrix with shape [num_nodes, num_node_features]
### The same the event representation - num_particles, num_particle_feature
### Read chunk of the events in paralel
### Select promt from DY, fakes from WJ
### Pt Reweighting

FEATURES=[
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
    if "WJ" in file_name:
        index = np.where(df['jet_match']==0)[0]
    elif "DY" in file_name:
        index = np.where(df['jet_match']==1)[0]
    index =random.choice(index)
    return index


class TauIdDataset(InMemoryDataset):
    
    def __init__(self, root, mode='test',transform=None, pre_transform=None):
        """

        :param root:
        :param mode:
        :param transform:
        :param pre_transform:
        """
        self.filenames = glob(root+"DY*.root") +  glob(root+"WJ*.root")
        random.shuffle(self.filenames) #shuffle the filenames
        self.root = root
        self.len = len(self.filenames)

    @property
    def raw_file_names(self):
        return self.filenames
    
    @property
    def processed_file_names(self):
        return ['training.pt', 'test.pt']
    
    
    def __getitem__(self, index):
        """
        """
        try:
            #print("file names: ", self.filenames[index])
            root_file = uproot.open(self.filenames[index])['Candidates']
            data = self.process(root_file, file_name=self.filenames[index])
            return data
        except Exception:
            pass

    def __len__(self):
        """
        """
        return self.len
    
    def process(self, root_file, file_name):
        """
        """
        data_list = []
        entrystart = index_choice(root_file, file_name)
        df = root_file.iterate(entrystart=entrystart, entrystop=entrystart+1)
        df = df.next()
        max_event = 60
        # Split the Features and Labels
        val = np.zeros((max_event, len(FEATURES)))
        for feature_index in range(len(FEATURES)):
            feature = FEATURES[feature_index]
            try:
                nCand = len(df[feature][0])
            except Exception:
                nCand = 1
            df[feature]=np.pad(df[feature], [(0, 60-nCand) ], mode='constant')

        x = torch.tensor(np.array([df[feature] for feature in FEATURES]).T, dtype=torch.float)
        data  = Data() #, edge_index=edge_index)
        data.y = torch.tensor(df['lepTauMatch_1'],  dtype=torch.int64)
        data.pos = data.x
        data_list.append(data)
        return data.x, data.pos, data.y
        

batch_size=100
tauId_dataset = TauIdDataset(TRAIN_SET)
loader = D.DataLoader(tauId_dataset, batch_size=100, shuffle=False, num_workers=0)
dataiter = iter(loader)


data_list = []
for batch in dataiter:
    for i in range(32):
        data_list.append(Data(x = batch[0][i], pos=batch[1][i], y = batch[2][i]))
    break

train_loader = DataLoader(data_list, batch_size=16, shuffle=True)
#### Definition of the Net
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
  
        self.conv1 = GCNConv(40, 10)
        self.conv2 = GCNConv(10, 32)
        self.conv3 = GCNConv(32, 1)
        self.dense = nn.Linear(1, 1)
        
    def forward(self, data):
        #, data.edge_index
        edge_index = knn_graph(data.pos, 4, loop=False)
        #edge_index = radius_graph(data.pos, loop=False, max_num_neighbors=4)
        #print("Edge Index:", edge_index.shape)
        x = nn.functional.relu(self.conv1(data.x, edge_index))
        print("X shape", x.shape)
        #x = global_max_pool(x, torch.tensor([1 for i in range(60)]))
        x = nn.functional.relu(self.conv2(x, edge_index))
        print("X 2 Convo Shape", x.shape)
        x = nn.functional.relu(self.conv3(x, edge_index))
        print("X 3 Convo Shape", x.shape)
        
        x = self.dense(x)
        print("X shape", x.shape)
        res =  nn.functional.log_softmax(x, dim=1)
        print("Res Shape:", res.shape)
        return res
#https://pytorch-geometric.readthedocs.io/en/latest/modules/data.html#torch_geometric.data.Batch

net = Net()
criterion = nn.BCELoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


def MLP(channels):
    return Seq(*[
        Seq(Lin(channels[i - 1], channels[i]), ReLU(), BN(channels[i]))
        for i in range(1, len(channels))
    ])

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GCNConv(40, 3)
        self.conv2 = GCNConv(5, 5)
        self.mlp1 = MLP([3, 3, 5, 5])
        self.mlp2 = MLP([5, 5, 3])
        self.line1 = MLP([5 + 3, 32])
        self.mlp = Seq(
            MLP([32, 16]), Dropout(0.5), MLP([16, 128]), Dropout(0.5),
            Lin(128, 2))
        
    def forward(self, data):
        x, edge_index = data.pos, None
        '''
        Because GCN needs to calculate the adjacency matrix.
        The idea is to build a graph with a patch, so our matrix built through local point clouds is actually very
        small and not as difficult to calculate as a large graph.
        '''
        edge_index = knn_graph(x, 4, data.batch)
	
        x1 = self.conv1(x, edge_index)
        
	print("After GCN size:", x1.size())
        x1 = self.mlp1(x1)
        x1 = F.relu(x1)
        x2 = self.conv2(x1, edge_index)
        
	print("After GCN size:", x2.size())
        x2 = self.mlp2(x2)
        
	print("After GCN size:", x.size())
        x2 = F.relu(x2)
        
	print(x2.shape)
        x4 = torch.cat([x1, x2], dim=1)
        
	print(x4.shape)
        out = self.line1(x4)

	print("Out: ", out.shape)
        out = global_mean_pool(out, data.batch, size=16)
        
	print(out.shape)
        out = self.mlp(out)
        return F.log_softmax(out, dim=1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epoch=0
for data in train_loader:
    epoch+=1
    inputs, labels = data, data.y
    print(labels.detach().numpy())
    print("Input shape", inputs.x.shape)
    print("Labels shape", labels.shape)
    optimizer.zero_grad()
    print("Num of features:",  data.num_features)
    outputs = model(data)
    print("Outputs: ", outputs.shape)
    target = data.y
    print("Target:", target.shape)
    loss = F.nll_loss(outputs, data.y)
    loss.backward()
    optimizer.step()
    print('Finished Training:', loss)
    #Accuracy
    outputs = (outputs>0.5).float()
    correct = (outputs == labels).float().sum()
    print("Epoch {}/{}, Loss: {:.3f}, Accuracy: {:.3f}".format(epoch+1,num_epochs, loss.data[0], correct/labels.shape[0]))
