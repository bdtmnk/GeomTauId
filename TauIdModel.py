""""
Model Definition and Training Process Specification:
""""


from torch_geometric.data import Dataset, Data
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch
import torch.nn.functional as F
from torch.nn import Sequential as S, Linear as L, ReLU, BatchNorm1d as BN, Dropout
from torch_geometric.nn import GCNConv, knn_graph, global_mean_pool, global_max_pool, GATConv, PointConv, TopKPooling, GlobalAttention, EdgeConv
from torch_geometric.nn.conv import MessagePassing

from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score
from pandas.api.types import CategoricalDtype



#Network Definition:

class EdgeConv(MessagePassing):
    def __init__(self, nn, aggr='mean', **kwargs):
        super(EdgeConv, self).__init__(aggr=aggr, **kwargs)
        self.nn = nn
    def forward(self, x, edge_index):
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        return self.propagate(edge_index, x=x)
    def message(self, x_i, x_j):
        return self.nn(torch.cat([x_i, x_j - x_i], dim=1))
    def __repr__(self):
        return '{}(nn={})'.format(self.__class__.__name__, self.nn)


def MLP(channels):
    """
    Create MLP with specified shape.
    :param channels: Shape of the MLP - list of the format [input size, first hidden layer size, ..., Nth hidden layer size, output size]
    :return:  MLP object
    """
    return S(*[ S(L(channels[i - 1], channels[i]), ReLU(), BN(channels[i])) for i in range(1, len(channels)) ])


class ResMLP(torch.nn.Module):
    """Class of MLP with residual connection."""

    def __init__(self, dim, depth):
        """
            Create MLP with residual connection of specified shape.
            :param dim: Number of nodes in each layer (will be the same for all the layers)
            :param depth: Number of layers in MLP
            """
        super(ResMLP, self).__init__()
        self.layers = []
        for i in range(depth):
            self.layers.append(S(L(dim, dim), ReLU(), BN(dim)))

    def forward(self, x):
        x_res  = x
        for layer in self.layers:
           x = layer(x)
        return x + x_res



class ECN3(torch.nn.Module):
    """
    Network with three EdgeConv layer and separated features.
    Features describing the event in general are fed directly into the MLP classifier.
    """
    def __init__(self, config):
        super(ECN3, self).__init__()
        self.conv1 = EdgeConv(MLP([72, 128]), aggr='mean')
        self.conv2 = EdgeConv(MLP([256, 256]), aggr='mean')
        self.conv3 = EdgeConv(MLP([512, 512]), aggr='mean')

        self.classifier = MLP([512, 512, 256])
        self.classifier1 = MLP([256, 128, 4])

        self.KNN_Number = config['KNN_Number']
    def forward(self, data):
        KNN_Number = self.KNN_Number
        x = data.x
        pos = data.pos
        batch = data.batch
        edge_index = knn_graph(pos, KNN_Number, batch, loop=True)
        x1 = self.conv1(x, edge_index)
        edge_index = knn_graph(x1, KNN_Number, batch, loop=True)
        x1 = self.conv2(x1, edge_index)
        edge_index = knn_graph(x1, KNN_Number, batch, loop=True)
        x1 = self.conv3(x1, edge_index)

        x1 = global_mean_pool(x1, batch)#, size=data.num_graphs)
        #x2 = torch.cat((x1, x), dim=1)
        x1 = self.classifier(x1)
        out = self.classifier1(x1)

        return F.softmax(out, dim=1)#torch.softmax(out, dim=4)#[:, 0]




"""
Training Process:
"""
# Prepare train data
TEST_PATH = "/beegfs/desy/user/dydukhle/TauId/samples/2018/train/"

start = time.time()
train_dataset = TauIdDataset( TEST_PATH, num=5000, nfiles=10, processes='all')
train_length = train_dataset.len
end = time.time() - start
print("Duration: ", end)


from torch_geometric.data import DataLoader
train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True, num_workers=8)
train_dataset.len =len(train_dataset.data)

from IPython.display import clear_output
import torchcontrib


## Oversample the train sample
def _overbalance(dump, mode='train', label='Y'):
    """
    Return Oversampled dataset
    """
    
    vc = dump[label].value_counts()
    classes = vc.keys()
    values = vc.values
    ov_b = []
    for _class in classes[1:]:
        df_class_1_over = dump[dump[label]==_class].sample(values[0], replace=True)
        ov_b.append(df_class_1_over)
    df_over = pd.concat(ov_b+ [dump[dump[label]==classes[0]]], axis=0)
    df_over = df_over.sample(frac=1)

    return df_over

