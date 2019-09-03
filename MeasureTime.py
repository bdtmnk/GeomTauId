import time

import torch
from torch_geometric.data import DataLoader

from LoadDataLegacy import TauIdDataset

TRAIN_SET = "/nfs/dust/cms/user/dydukhle/TauIdSamples/TauId/2016/train_samples/"
TEST_SET = "/nfs/dust/cms/user/dydukhle/TauIdSamples/TauId/2016/test_samples/"
TRAINING_RES = "/nfs/dust/cms/user/bukinkir/TauId/ECN/"
epoch = 50
batch_size = 2048
num = 2048


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
    start_time = time.time()
    test_dataset = TauIdDataset(TEST_SET, num=num)
    time_dataset = time.time()
    test_length = test_dataset.len
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=1)
    time_loader = time.time()

    net, _, epoch, _ = load_model('{0}ECN_{1}.pt'.format(TRAINING_RES, epoch))
    print(net)

    time_model = time.time()

    for data in iter(test_loader):
        inputs, labels = data.x, data.y

        time_starteval = time.time()
        outputs = net(data)
        time_end = time.time()
        target = labels.detach().numpy()
        file = open("{0}timing.log".format(TRAINING_RES), 'w')
        file.write("Batch size: {0}; Dataset initialization: {1}; Dataloader initialization: {2}; Model loading: {3};  Data loading: {4};  Propagation: {5}".format(batch_size,
                                                                                                       time_dataset - start_time,time_loader - time_dataset, time_model - time_loader, time_starteval - time_model, time_end - time_starteval))
        file.close()
