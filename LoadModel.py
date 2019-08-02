import torch
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, BatchNorm1d as BN, Dropout
from torch_geometric.nn import GCNConv, DynamicEdgeConv,  knn_graph, global_mean_pool
import torch.nn.functional as F


def MLP(channels):
    return Seq(*[
        Seq(Lin(channels[i - 1], channels[i]), ReLU(), BN(channels[i]))
        for i in range(1, len(channels))
    ])


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GCNConv(34, 3)
        self.conv2 = GCNConv(5, 5)
        self.mlp1 = MLP([3, 3, 5, 5])
        self.mlp2 = MLP([5, 5, 3])
        self.line1 = MLP([5 + 3, 32])
        self.mlp = Seq(
            MLP([32, 16]), Dropout(0.5), MLP([16, 128]), Dropout(0.5),
            Lin(128, 2))

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
        return F.log_softmax(out, dim=1)

# class Net(torch.nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         # self.mlp1 = MLP([68, 64])
#         # self.mlp2 = MLP([128, 128])
#         self.conv1 = DynamicEdgeConv(MLP([68, 64]), 3)
#         self.conv2 = DynamicEdgeConv(MLP([128, 128]), 3)
#         self.mlp = Seq(
#             MLP([128, 256]), Dropout(0.5), MLP([256, 128]), Dropout(0.5),
#             Lin(128, 2))
#
#     def forward(self, data):
#         x, edge_index = data.pos, None
#         print("input size:",x.size())
#         print("data batch", len(data.batch))
#         '''
#         Because GCN needs to calculate the adjacency matrix.
#         The idea is to build a graph with a patch, so our matrix built through local point clouds is actually very
#         small and not as difficult to calculate as a large graph.
#         '''
#         print('start')
#         # edge_index = knn_graph(x, 4, data.batch)
#         x1 = self.conv1(x, data.batch)
#         print('conv1')
#         x2 = self.conv2(x1, data.batch)
#         print('conv2')
#         out = global_mean_pool(x2, data.batch, size=data.num_graphs)
#         print('pool')
#         out = self.mlp(out)
#         print('mlp')
#         return F.log_softmax(out, dim=1)
