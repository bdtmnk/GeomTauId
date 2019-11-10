import time
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import roc_auc_score 
from LoadData import TauIdDataset, get_weights
from LoadModel import ECN2, ECN3
from Logger import Logger


BASE_PATH = "/beegfs/desy/user/dydukhle/TauId/GeomTauId/GeomTauId/"
TRAIN_SET = "{0}/TauIdSamples/train_samples/".format(BASE_PATH)
TEST_SET = "{0}/TauIdSamples/train_samples/".format(BASE_PATH)
TRAINING_RES = "{0}/GeomTauId/Results/GCN_DM_KNN_10_ECN3/".format(BASE_PATH)
RESUME_TRAINING = False
EPOCH = 0
nFiles = 1
nEvents = 50#000
nBatch = 10#00
make_test = False
use_cuda = torch.cuda.is_available()
device = torch.device('cpu')

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
    train_dataset = TauIdDataset(TRAIN_SET, num=nEvents, nfiles=nFiles)
    train_length = train_dataset.len
    train_loader = DataLoader(train_dataset, batch_size=nBatch, shuffle=True, num_workers=16)

    train = time.time()
    print("Train data loaded: {0}".format(train - start))

    # Prepare test data
    test_dataset = TauIdDataset(TEST_SET, num=nEvents, nfiles=nFiles)
    test_length = test_dataset.len
    test_loader = DataLoader(test_dataset, batch_size=nBatch, shuffle=True, num_workers=16)
    writer = SummaryWriter()

    test = time.time()
    print("Test data loaded: {0}".format(test - train))
    KNN_Number = 10#4
    # Load or create the network
    if RESUME_TRAINING:
        net, optimizer, _, scheduler = load_model('{0}ECN_{1}.pt'.format(TRAINING_RES, EPOCH))
        print ("Model loaded")
    else:
        net = ECN3(KNN_Number=KNN_Number).to(device)
        optimizer = torch.optim.SGD(net.parameters(), lr=0.5, momentum=0.9)
        EPOCH = 0

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 100)
    log = Logger(TRAINING_RES, 'ECN',  RESUME_TRAINING)
    torch.set_num_threads(1)

    for epoch in range(EPOCH + 1, 10):
	
        for data in iter(train_loader):
            data = data.to(device)
            inputs, labels = data.x, data.y.type(torch.FloatTensor)
            # Compute the weights of events
            pt_train = inputs.cpu().detach().numpy()[:, 3]
            _, ind = np.unique(data.batch.cpu().detach().numpy(), return_index=True)
            pt_train = pt_train[ind]
            Y = labels.cpu().detach().numpy()
            weight = get_weights(pt_train, Y)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(data)
            loss = F.binary_cross_entropy(outputs, labels, weight=weight)
            loss.backward()
            optimizer.step()
            log.eval_train(loss, labels, outputs)
        scheduler.step()
        n_iter = epochs
        accuracy_train = roc_auc_score(labels.cpu().detach().numpy(), outputs.cpu().detach().numpy())
        accuracy_test = roc_auc_score(labels.cpu().detach().numpy(), outputs.cpu().detach().numpy())
        writer.add_scalar('Loss/train', loss.cpu().detach().numpy(), n_iter)
        writer.add_scalar('Loss/test', loss.cpu().detach().numpy(), n_iter)
        writer.add_scalar('Accuracy/train', accuracy_train, n_iter)
        writer.add_scalar('Accuracy/test', accuracy_test, n_iter)
        torch.save({
            'epoch': epoch,
            'net': net,
            'optimizer': optimizer,
            'loss': loss,
            'scheduler': scheduler
        }, '{0}ECN_{1}.pt'.format(TRAINING_RES, epoch))
        # Run validation
        
        if make_test==True and epoch % 1 == 0:
            with torch.no_grad():
                labels_list = []
                outputs_list = []
                loss_list = []
                for data in iter(test_loader):
                    #data = data.to(device)
                    inputs, labels = data.x, data.y.type(torch.FloatTensor)
                    
                    outputs = net(data)
                    labels_list.append(labels)
                    outputs_list.append(outputs)
                    loss_list.append(F.binary_cross_entropy(outputs, labels))
        
        log.eval_test(epoch, loss_list, labels_list, outputs_list)
        
    log.close()
    writer.close()
    print('Finished Training')
