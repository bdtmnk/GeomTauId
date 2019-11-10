import argparse
import os
import time
import pandas as pd
import torch
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
from torch_geometric.data import DataLoader
from LoadData import TauIdDataset, get_weights
from LoadModel import ECN2
from Logger import Logger

#TODO add tensorboard:

TRAIN_SET = "/nfs/dust/cms/user/dydukhle/GeomTauID/GeomTauId/TauIdSamples/train_samples/"
TEST_SET = "/nfs/dust/cms/user/dydukhle/GeomTauID/GeomTauId/TauIdSamples/test_samples/"
TRAINING_RES = "/nfs/dust/cms/user/dydukhle/GeomTauID/.GCN_DM*_QCD_6k_ECN3/"
#.GCN_DM*_QCD/"
TEST_RES = "/nfs/dust/cms/user/dydukhle/GeomTauID/.GCN_DM*_QCD_6k/TEST_RES_"
#TEST_RES = "/nfs/dust/cms/user/dydukhle/GeomTauID/.GCN_DM*_QCD_6k_ECN3/TEST_RES"

parser = argparse.ArgumentParser()
parser.add_argument("--file", default="QCD_2016_85_tree.root", help="Root file to Evaluate")


args = parser.parse_args()
file_to_evaluate = args.file


#DATASET TO Evaluate


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


if __name__ == "__main__":
    start = time.time()

    # Prepare test data
    test_dataset = TauIdDataset(TEST_SET, num=1000, mode="test")#, filename=file_to_evaluate)

    test_length = test_dataset.len
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=True,
                            num_workers=1)

    #len(arr['nPFCands_1'])
    # Load or create the network
    net, optimizer, _, scheduler = load_model('{0}/ECN_45.pt'.format(TRAINING_RES))

    torch.set_num_threads(1)

    labels_list = []
    outputs_list = []
    loss_list = []

    for data in iter(test_loader):

        inputs, labels = data.x, data.y.type(torch.FloatTensor)

        # Compute the weights of events
        pt_train = inputs.detach().numpy()[:, 3]
        _, ind = np.unique(data.batch, return_index=True)
        pt_train = pt_train[ind]
        Y = labels.detach().numpy()
        weight = get_weights(pt_train, Y)
        with torch.no_grad():
            # forward + backward + optimize
            outputs = net(data)
            outputs = outputs.detach().numpy()	
	    print("Outputs: ", outputs.shape)
	    print("Labels: ", labels.shape)
	    print("Weights: ", weight.shape)
#            loss = F.binary_cross_entropy(outputs, labels, weight=weight)
#            print("Loss:", loss)

            accuracy = np.where((Y + outputs) / 2 != 0.5)[0].shape[0]/Y.shape[0]
            print("Accuracy:", accuracy)

            #Fill the historgrams;
            summary = pd.DataFrame()

            summary['labels'] = Y
            summary['output'] = outputs#.detach().numpy()
	    summary['pt'] = inputs.detach().numpy()[:, 3]
	    summary['dm'] = np.argmax(inputs.detach().numpy()[:, 34:39])
            #Get DM data;
            #Get pt ;
            #Get size of graph;
            if os.path.exists(TEST_RES):
                summary.to_csv(TEST_RES + "/{0}.csv".format(file_to_evaluate))
            else:
                os.mkdir(TEST_RES)
	    break
        #Effieciency plots;

        #Pt - Dep;

        #For different datasets;


#        labels_list.append(labels)
#        outputs_list.append(outputs)
#        loss_list.append(F.binary_cross_entropy(outputs, labels))
        print('Finished Training')

