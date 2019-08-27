import torch
from torch.nn import Sequential as S, Linear as L, ReLU, BatchNorm1d as BN, Dropout, ELU, Conv1d
from torch_geometric.nn import GCNConv,  knn_graph, global_mean_pool, global_max_pool,  Reshape, GATConv, PointConv, TopKPooling, GlobalAttention
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
# from torch_geometric.nn.pool import EdgePooling
from math import ceil


class EdgeConv(MessagePassing):

    def __init__(self, nn, aggr='mean', **kwargs):
        super(EdgeConv, self).__init__(aggr=aggr, **kwargs)
        self.nn = nn

    def forward(self, x, edge_index):
        """"""
        x = x.unsqueeze(-1) if x.dim() == 1 else x

        return self.propagate(edge_index, x=x)

    def message(self, x_i, x_j):
        return self.nn(torch.cat([x_i, x_j - x_i], dim=1))

    def __repr__(self):
        return '{}(nn={})'.format(self.__class__.__name__, self.nn)


def MLP(channels):
    return S(*[
        S(L(channels[i - 1], channels[i]), ReLU(), BN(channels[i]))
        for i in range(1, len(channels))
    ])


class ResMLP(torch.nn.Module):
    def __init__(self, dim, depth):
        super(ResMLP, self).__init__()
        self.layers = []
        for i in range(depth):
            self.layers.append(S(L(dim, dim), ReLU(), BN(dim)))

    def forward(self, x):
        x_res  = x
        for layer in self.layers:
           x = layer(x)
        return x + x_res


# Graph Convolution Network (don't work)
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GCNConv(40, 3)
        self.conv2 = GCNConv(5, 5)
        self.mlp1 = MLP([3, 3, 5, 5])
        self.mlp2 = MLP([5, 5, 3])
        self.line1 = MLP([5 + 3, 32])
        self.mlp = S(
            MLP([32, 16]), Dropout(0.5), MLP([16, 128]), Dropout(0.5),
            L(128, 1))

    def forward(self, data):
        x, edge_index = data.x, None
        pos = data.pos
        print("input size:", x.size())
        print("data batch", len(data.batch))
        '''
        Because GCN needs to calculate the adjacency matrix.
        The idea is to build a graph with a patch, so our matrix built through local point clouds is actually very
        small and not as difficult to calculate as a large graph.
        '''
        edge_index = knn_graph(pos, 4, data.batch)
        # print(x)
        # print(data.y)
        x1 = self.conv1(x, edge_index)
        print("After GCN size:", x1.size())
        x1 = self.mlp1(x1)
        x1 = F.relu(x1)
        x2 = self.conv2(x1, edge_index)
        print("After GCN size:", x2.size())
        x2 = self.mlp2(x2)
        print("After GCN size:", x2.size())
        x2 = F.relu(x2)
        print(x2.shape)
        x4 = torch.cat([x1, x2], dim=1)
        print(x4.shape)
        out = self.line1(x4)
        print("Out: ", out.shape)
        out = global_mean_pool(out, data.batch, size=data.num_graphs)
        out = self.mlp(out)
        # return F.log_softmax(out, dim=1)
        return torch.sigmoid(out)[:, 0]


# Simple, 1-layer network
class ECN(torch.nn.Module):
    def __init__(self):
        super(ECN, self).__init__()
        self.conv = EdgeConv(MLP([118, 128]), aggr='mean')
        self.classifier = MLP([128, 256, 1])

    def forward(self, data):
        x = data.x
        pos = data.pos
        batch = data.batch
        edge_index = knn_graph(pos, 3, batch, loop=False)
        print('index')
        x1 = self.conv(x, edge_index)
        print('conv')
        x2 = global_mean_pool(x1, batch, size=data.num_graphs)
        print('pool')
        out = self.classifier(x2)
        print('mlp')
        return torch.sigmoid(out)[:, 0]


# Deeper, 3-layer network
class ECN2(torch.nn.Module):
    def __init__(self):
        super(ECN2, self).__init__()
        # self.conv1 = EdgeConv(MLP([118, 128]), aggr='mean')
        # self.conv2 = EdgeConv(MLP([256, 256]), aggr='mean')
        # self.conv3 = EdgeConv(MLP([512, 512]), aggr='mean')
        self.conv1 = EdgeConv(MLP([106, 128, 128]), aggr='mean')
        self.conv2 = EdgeConv(MLP([256, 256, 256]), aggr='mean')
        self.conv3 = EdgeConv(MLP([512, 512, 512]), aggr='mean')
        self.classifier = MLP([512, 512, 1])

    def forward(self, data):
        x = data.x
        pos = data.pos
        batch = data.batch
        edge_index = knn_graph(pos, 4, batch, loop=False)
        x1 = self.conv1(x, edge_index)
        # print('conv1')
        edge_index = knn_graph(x1, 4, batch, loop=False)
        x1 = self.conv2(x1, edge_index)
        # print('conv2')
        edge_index = knn_graph(x1, 4, batch, loop=False)
        x1 = self.conv3(x1, edge_index)
        # print('conv3')
        x2 = global_mean_pool(x1, batch, size=data.num_graphs)
        # print('pool')
        out = self.classifier(x2)
        # print('mlp')
        return torch.sigmoid(out)[:, 0]


# Network with separated features
class ECN3(torch.nn.Module):
    def __init__(self):
        super(ECN3, self).__init__()
        self.conv1 = EdgeConv(MLP([102, 64]), aggr='mean')
        self.conv2 = EdgeConv(MLP([128, 128]), aggr='mean')
        self.conv3 = EdgeConv(MLP([256, 256]), aggr='mean')
        # self.mlp = MLP([8, 8, 8])
        self.classifier = MLP([264, 512, 1])

    def forward(self, data):
        x = data.x
        pos = data.pos
        x2 = data.x2
        batch = data.batch
        edge_index = knn_graph(pos, 3, batch, loop=False)
        x1 = self.conv1(x, edge_index)
        print('conv1')
        edge_index = knn_graph(x1, 3, batch, loop=False)
        x1 = self.conv2(x1, edge_index)
        print('conv2')
        edge_index = knn_graph(x1, 3, batch, loop=False)
        x1 = self.conv3(x1, edge_index)
        print('conv3')
        x1 = global_mean_pool(x1, batch, size=data.num_graphs)
        # x2 = self.mlp(x2)
        x2 = torch.cat((x1, x2), dim=1)
        print('pool')
        out = self.classifier(x2)
        print('mlp')
        return torch.sigmoid(out)[:, 0]


# ResNet
class ECN4(torch.nn.Module):
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
        edge_index = knn_graph(pos, 3, batch, loop=False)
        x1 = self.conv1(x, edge_index)
        print('conv1')
        edge_index = knn_graph(x1, 3, batch, loop=False)
        x1 = self.conv2(x1, edge_index)
        print('conv2')
        edge_index = knn_graph(x1, 3, batch, loop=False)
        x1 = self.conv3(x1, edge_index)
        print('conv3')
        x2 = global_mean_pool(x1, batch, size=data.num_graphs)
        print('pool')
        out = self.classifier(x2)
        print('mlp')
        return torch.sigmoid(out)[:, 0]


# DenseNet
class ECN5(torch.nn.Module):
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
        edge_index = knn_graph(pos, 4, batch, loop=False)
        x = self.conv1(x, edge_index)
        print('conv1')
        edge_index = knn_graph(x, 4, batch, loop=False)
        x1 = self.conv2(x, edge_index)
        x = torch.cat((x, x1), dim=1)
        print('conv2')
        edge_index = knn_graph(x1, 4, batch, loop=False)
        x1 = self.conv3(x, edge_index)
        x = torch.cat((x, x1), dim=1)
        print('conv3')
        edge_index = knn_graph(x1, 4, batch, loop=False)
        x1 = self.conv4(x, edge_index)
        x = torch.cat((x, x1), dim=1)
        print('conv4')
        edge_index = knn_graph(x1, 4, batch, loop=False)
        x1 = self.conv5(x, edge_index)
        x = torch.cat((x, x1), dim=1)
        print('conv5')
        x2 = global_mean_pool(x, batch, size=data.num_graphs)
        print('pool')
        out = self.classifier(x2)
        print('mlp')
        return torch.sigmoid(out)[:, 0]


# Network with pooling
class ECN6(torch.nn.Module):
    def __init__(self):
        super(ECN6, self).__init__()
        self.conv1 = EdgeConv(MLP([106, 128]), aggr='mean')
        self.conv2 = EdgeConv(MLP([256, 256]), aggr='mean')
        self.conv3 = EdgeConv(MLP([512, 512]), aggr='mean')
        self.pool1 = TopKPooling(128, 0.8)
        self.pool2 = TopKPooling(256, 0.8)
        self.classifier = MLP([512, 512, 1])

    def forward(self, data):
        x = data.x
        pos = data.pos
        batch = data.batch
        edge_index = knn_graph(pos, 4, batch, loop=False)
        x1 = self.conv1(x, edge_index)
        print('conv1')
        x1, edge_index,  _, batch, _= self.pool1(x1, edge_index, batch=batch)
        print('pool1')
        # edge_index = knn_graph(x1, 4, batch, loop=False)
        x1 = self.conv2(x1, edge_index)
        print('conv2')
        x1, edge_index, _, batch, _ = self.pool2(x1, edge_index, batch=batch)
        print('pool2')
        # edge_index = knn_graph(x1, 4, batch, loop=False)
        x1 = self.conv3(x1, edge_index)
        print('conv3')
        x2 = global_mean_pool(x1, batch, size=data.num_graphs)
        print('pool')
        out = self.classifier(x2)
        print('mlp')
        return torch.sigmoid(out)[:, 0]


# Another network with pooling
class ECN7(torch.nn.Module):
    def __init__(self):
        super(ECN7, self).__init__()
        self.conv1 = EdgeConv(MLP([106, 128]), aggr='mean')
        self.conv2 = EdgeConv(MLP([256, 256]), aggr='mean')
        self.conv3 = EdgeConv(MLP([512, 512]), aggr='mean')
        self.GlobAtt = GlobalAttention(MLP([512, 256, 1]), MLP([512, 512]))
        self.classifier = MLP([512, 512, 1])

    def forward(self, data):
        x = data.x
        pos = data.pos
        batch = data.batch
        edge_index = knn_graph(pos, 4, batch, loop=False)
        x1 = self.conv1(x, edge_index)
        print('conv1')
        edge_index = knn_graph(x1, 4, batch, loop=False)
        x1 = self.conv2(x1, edge_index)
        print('conv2')
        edge_index = knn_graph(x1, 4, batch, loop=False)
        x1 = self.conv3(x1, edge_index)
        print('conv3')
        x2 = global_mean_pool(x1, batch, size=data.num_graphs)
        print('pool')
        out = self.classifier(x2)
        print('mlp')
        return torch.sigmoid(out)[:, 0]


class XCN(torch.nn.Module):
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


class GATN(torch.nn.Module):
    def __init__(self):
        super(GATN, self).__init__()
        self.conv = GATConv(59, 128)
        self.classifier = MLP([128, 256, 1])

    def forward(self, data):
        x = data.x
        pos = data.pos
        batch = data.batch
        edge_index = knn_graph(pos, 3, batch, loop=False)
        print('index')
        x1 = self.conv(x, edge_index)
        print('conv')
        x2 = global_mean_pool(x1, batch, size=data.num_graphs)
        print('pool')
        out = self.classifier(x2)
        print('mlp')
        return torch.sigmoid(out)[:, 0]


class PCN(torch.nn.Module):
    def __init__(self):
        super(PCN, self).__init__()
        self.conv =PointConv(MLP([62, 128]), MLP([128, 128]))
        self.classifier = MLP([128, 256, 1])

    def forward(self, data):
        x = data.x
        pos = data.pos
        batch = data.batch
        edge_index = knn_graph(pos, 3, batch, loop=False)
        print('index')
        x1 = self.conv(x, pos,  edge_index)
        print('conv')
        x2 = global_mean_pool(x1, batch, size=data.num_graphs)
        print('pool')
        out = self.classifier(x2)
        print('mlp')
        return torch.sigmoid(out)[:, 0]
