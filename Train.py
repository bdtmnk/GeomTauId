import time

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import DataLoader

from LoadData import TauIdDataset, get_weights
from LoadModel import ECN2
from Logger import Logger

TRAIN_SET = "/nfs/dust/cms/user/dydukhle/TauIdSamples/TauId/2016/train_samples/"
TEST_SET = "/nfs/dust/cms/user/dydukhle/TauIdSamples/TauId/2016/test_samples/"
TRAINING_RES = "/nfs/dust/cms/user/bukinkir/TauId/ECN9/"
RESUME_TRAINING = False
EPOCH = 0


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


if __name__ == "__main__":
    start = time.time()

    # Prepare train data
    train_dataset = TauIdDataset(TRAIN_SET, num=5000)
    train_length = train_dataset.len
    train_loader = DataLoader(train_dataset, batch_size=1000, shuffle=True, num_workers=1)

    train = time.time()
    print("Train data loaded: {0}".format(train - start))

    # Prepare test data
    test_dataset = TauIdDataset(TEST_SET, num=5000)
    test_length = test_dataset.len
    test_loader = DataLoader(test_dataset, batch_size=5000, shuffle=True, num_workers=1)

    test = time.time()
    print("Test data loaded: {0}".format(test - train))

    # Load or create the network
    if RESUME_TRAINING:
        net, optimizer, _, scheduler = load_model('{0}ECN_{1}.pt'.format(TRAINING_RES, EPOCH))
        print ("Model loaded")
    else:
        net = ECN2()
        optimizer = torch.optim.SGD(net.parameters(), lr=0.5, momentum=0.9)
        EPOCH = 0

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 100)
    log = Logger(TRAINING_RES, 'ECN',  RESUME_TRAINING)
    torch.set_num_threads(1)

    for epoch in range(EPOCH + 1, 101):
        for data in iter(train_loader):

            inputs, labels = data.x, data.y.type(torch.FloatTensor)

            # Compute the weights of events
            pt_train = inputs.detach().numpy()[:, 3]
            _, ind = np.unique(data.batch, return_index=True)
            pt_train = pt_train[ind]
            Y = labels.detach().numpy()
            weight = get_weights(pt_train, Y)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(data)
            loss = F.binary_cross_entropy(outputs, labels, weight=weight)
            loss.backward()
            optimizer.step()
            print(loss)
            log.eval_train(loss, labels, outputs)
        scheduler.step()

        log.save_train(epoch)
        torch.save({
            'epoch': epoch,
            'net': net,
            'optimizer': optimizer,
            'loss': loss,
            'scheduler': scheduler
        }, '{0}ECN_{1}.pt'.format(TRAINING_RES, epoch))

        # Run validation
        if epoch % 10 == 0:
            with torch.no_grad():
                labels_list = []
                outputs_list = []
                loss_list = []
                for data in iter(test_loader):
                    inputs, labels = data.x, data.y.type(torch.FloatTensor)
                    outputs = net(data)
                    labels_list.append(labels)
                    outputs_list.append(outputs)
                    loss_list.append(F.binary_cross_entropy(outputs, labels))
            log.eval_test(epoch, loss_list, labels_list, outputs_list)
    log.close()
    print('Finished Training')
