import torch
import torch.nn.functional as F
from torch.nn import Sequential as S, Linear as L, ReLU, BatchNorm1d as BN, Dropout
from torch_geometric.nn import GCNConv, knn_graph, global_mean_pool, GATConv, PointConv, TopKPooling, GlobalAttention, EdgeConv
from torch_geometric.nn.conv import MessagePassing


config=["isReidual", "KNN_Number", "MLP"]


def load_model(path):
    """
    Load model from .pt file.
    :param path: Path to the .pt file where the model is stored
    :return: Network, optimizer, number of epoch when the network was stored, LR scheduler
    """
    checkpoint = torch.load(path)
    net = checkpoint['net']
    optimizer = checkpoint['optimizer']
    epoch = checkpoint['epoch']
    scheduler = checkpoint['scheduler']
    return net, optimizer, epoch, scheduler


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




class ECN(torch.nn.Module):
    """Network with one EdgeConv layer"""

    def __init__(self, config):
        super(ECN, self).__init__()
        self.conv = EdgeConv(MLP([118, 128]), aggr='mean')
        self.classifier = MLP([128, 256, 1])
        self.KNN_Number = config['KNN_Number']

    def forward(self, data):
        x = data.x
        pos = data.pos
        batch = data.batch

        edge_index = knn_graph(pos, KNN_Number, batch, loop=False)
        x1 = self.conv(x, edge_index)
        x2 = global_mean_pool(x1, batch, size=data.num_graphs)
        out = self.classifier(x2)
        return torch.sigmoid(out)[:, 0]


class ECN2(torch.nn.Module):
    """Network with three EdgeConv layers"""
    def __init__(self):
        super(ECN2, self).__init__()

        self.conv1 = EdgeConv(MLP([106, 128]), aggr='mean')
        self.conv2 = EdgeConv(MLP([256, 256]), aggr='mean')
        self.conv3 = EdgeConv(MLP([512, 512]), aggr='mean')
        self.classifier = MLP([512, 512, 1])

    def forward(self, data):
        x = data.x
        pos = data.pos
        batch = data.batch

        edge_index = knn_graph(pos, 6, batch, loop=False)
        x1 = self.conv1(x, edge_index)
        print('conv1')
        edge_index = knn_graph(x1, 6, batch, loop=False)
        x1 = self.conv2(x1, edge_index)
        print('conv2')
        edge_index = knn_graph(x1, 6, batch, loop=False)
        x1 = self.conv3(x1, edge_index)
        print('conv3')
        x2 = global_mean_pool(x1, batch, size=data.num_graphs)
        print('pool')
        out = self.classifier(x2)
        print('mlp')
        return torch.sigmoid(out)[:, 0]


class ECN3(torch.nn.Module):
    """
    Network with three EdgeConv layer and separated features.

    Features describing the event in general are fed directly into the MLP classifier.
    """
    def __init__(self, config):
        super(ECN3, self).__init__()
        self.conv1 = EdgeConv(MLP([70, 128]), aggr='mean')
        self.conv2 = EdgeConv(MLP([256, 256]), aggr='mean')
        self.conv3 = EdgeConv(MLP([512, 512]), aggr='mean')
        self.classifier = MLP([512, 512, 1])
        self.KNN_Number = config['KNN_Number']
    def forward(self, data):
        KNN_Number = self.KNN_Number
        x = data.x
        pos = data.pos
        batch = data.batch
        edge_index = knn_graph(pos, KNN_Number, batch, loop=False)
        x1 = self.conv1(x, edge_index)
        edge_index = knn_graph(x1, KNN_Number+2, batch, loop=False)
        x1 = self.conv2(x1, edge_index)
        edge_index = knn_graph(x1, KNN_Number+4, batch, loop=False)
        x1 = self.conv3(x1, edge_index)
        x1 = global_mean_pool(x1, batch, size=data.num_graphs)

        out = self.classifier(x1)
        return torch.sigmoid(out)[:, 0]




class ECN4(torch.nn.Module):
    """Network with three EdgeConv layers with residual connections"""
    def __init__(self):
        super(ECN4, self).__init__()
        self.conv1 = EdgeConv(S(MLP([118, 128]), ResMLP(128, 2)), aggr='mean')
        self.conv2 = EdgeConv(ResMLP(256, 3), aggr='mean')
        self.conv3 = EdgeConv(ResMLP(512, 3), aggr='mean')
        self.classifier = MLP([512, 512, 1])

    def forward(self, data):
        x = data.x
        pos = data.pos
        batch = data.batch

        edge_index = knn_graph(pos, 6, batch, loop=False)
        x1 = self.conv1(x, edge_index)
        print('conv1')
        edge_index = knn_graph(x1, 6, batch, loop=False)
        x1 = self.conv2(x1, edge_index)
        print('conv2')
        edge_index = knn_graph(x1, 6, batch, loop=False)
        x1 = self.conv3(x1, edge_index)
        print('conv3')
        x2 = global_mean_pool(x1, batch, size=data.num_graphs)
        print('pool')
        out = self.classifier(x2)
        print('mlp')
        return torch.sigmoid(out)[:, 0]


class ECN5(torch.nn.Module):
    """Network with five EdgeConv layers with dense connections"""
    def __init__(self):
        super(ECN5, self).__init__()
        self.conv1 = EdgeConv(MLP([106, 128]), aggr='mean')
        self.conv2 = EdgeConv(MLP([256, 128, 32]), aggr='mean')
        self.conv3 = EdgeConv(MLP([320, 128, 32]), aggr='mean')
        self.conv4 = EdgeConv(MLP([384, 256, 32]), aggr='mean')
        self.conv5 = EdgeConv(MLP([448, 256, 32]), aggr='mean')
        self.classifier = MLP([256, 512, 1])

    def forward(self, data):
        x = data.x
        pos = data.pos
        batch = data.batch

        edge_index = knn_graph(pos, 6, batch, loop=False)
        x = self.conv1(x, edge_index)
        print('conv1')
        edge_index = knn_graph(x, 6, batch, loop=False)
        x1 = self.conv2(x, edge_index)
        x = torch.cat((x, x1), dim=1)
        print('conv2')
        edge_index = knn_graph(x1, 6, batch, loop=False)
        x1 = self.conv3(x, edge_index)
        x = torch.cat((x, x1), dim=1)
        print('conv3')
        edge_index = knn_graph(x1, 6, batch, loop=False)
        x1 = self.conv4(x, edge_index)
        x = torch.cat((x, x1), dim=1)
        print('conv4')
        edge_index = knn_graph(x1, 6, batch, loop=False)
        x1 = self.conv5(x, edge_index)
        x = torch.cat((x, x1), dim=1)
        print('conv5')
        x2 = global_mean_pool(x, batch, size=data.num_graphs)
        print('pool')
        out = self.classifier(x2)
        print('mlp')
        return torch.sigmoid(out)[:, 0]


class XCN(torch.nn.Module):
    """Network with one XConv layer"""
    def __init__(self):
        super(XCN, self).__init__()
        self.conv = XConv(40, 128, 30, 4, 10)
        self.classifier = MLP([128, 256, 1])

    def forward(self, data):
        x = data.x
        pos = data.pos
        batch = data.batch
        print('index')
        x1 = self.conv(x, pos, batch)
        print('conv')
        x2 = global_mean_pool(x1, batch, size=data.num_graphs)
        print('pool')
        out = self.classifier(x2)
        print('mlp')
        return torch.sigmoid(out)[:, 0]


class PCN(torch.nn.Module):
    """Network with one PointConv layer"""
    def __init__(self):
        super(PCN, self).__init__()
        self.conv =PointConv(MLP([62, 128]), MLP([128, 128]))
        self.classifier = MLP([128, 256, 1])

    def forward(self, data):
        x = data.x
        pos = data.pos
        batch = data.batch
        edge_index = knn_graph(pos, 6, batch, loop=False)
        print('index')
        x1 = self.conv(x, pos,  edge_index)
        print('conv')
        x2 = global_mean_pool(x1, batch, size=data.num_graphs)
        print('pool')
        out = self.classifier(x2)
        print('mlp')
        return torch.sigmoid(out)[:, 0]
