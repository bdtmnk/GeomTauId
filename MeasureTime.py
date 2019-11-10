import time

import torch
from torch_geometric.data import DataLoader

from LoadData import TauIdDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_PATH = "/beegfs/desy/user/dydukhle/TauId/GeomTauId/GeomTauId/"
TRAIN_SET = "{0}/TauIdSamples/train_samples/".format(BASE_PATH)
TEST_SET = "{0}/TauIdSamples/train_samples/".format(BASE_PATH)
TRAINING_RES = "{0}/GeomTauId/Results/GCN_DM_KNN_ECN3/".format(BASE_PATH)
RESUME_TRAINING = False
EPOCH = 33
nFiles = 1
nEvents = 1
batch_size = 1
num = 10


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
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    start_time = time.time()
    test_dataset = TauIdDataset(TEST_SET, num=num)
    time_dataset = time.time()
    test_length = test_dataset.len
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=16)
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
