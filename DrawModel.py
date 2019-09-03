# Work in progress, script may not work
import torch
from torch_geometric.data import  Data
from torchviz import make_dot

TRAIN_SET = "/nfs/dust/cms/user/dydukhle/TauIdSamples/TauId/2016/train_samples/"
TEST_SET = "/nfs/dust/cms/user/dydukhle/TauIdSamples/TauId/2016/test_samples/"
TRAINING_RES = "/nfs/dust/cms/user/bukinkir/TauId/ECN3/"
epoch = 68
batch_size = 2048
num = 16384


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
    loss = checkpoint['loss']
    params_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print("Number of parameters: {0}".format(params_num))
    net.eval()
    return net, optimizer, epoch, loss


if __name__ == '__main__':
    data = Data()
    data.pos = torch.randn(8, 3)
    data.x = torch.randn(8, 59)

    net, _, epoch, _ = load_model('{0}ECN_{1}.pt'.format(TRAINING_RES, epoch))
    make_dot(net(data), params=dict(net.named_parameters()))
