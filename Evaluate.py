<<<<<<< HEAD
import time

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import DataLoader

from LoadData import TauIdDataset

TRAIN_SET = "/nfs/dust/cms/user/dydukhle/TauIdSamples/TauId/2016/train_samples/"
TEST_SET = "/nfs/dust/cms/user/dydukhle/TauIdSamples/TauId/2016/test_samples/"
TRAINING_RES = "/nfs/dust/cms/user/bukinkir/TauId/ECN6/"
epoch = 60
batch_size = 2048
num = 16384
=======
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
>>>>>>> 1490b45245b585f86aad3cc9b868aacb6e7ace19


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


<<<<<<< HEAD
if __name__ == '__main__':
    test_dataset = TauIdDataset(TEST_SET, num=num, mode='test')
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=1)

    net, _, epoch, _ = load_model('{0}ECN_{1}.pt'.format(TRAINING_RES, epoch))
    print(net)

    score = np.array([])
    target = np.array([])
    mva = np.array([])
    nLooseTaus = np.array([])
    nPFCands_1 = np.array([])
    decayMode_1 = np.array([])
    lepRecoPt_1 = np.array([])
    lepRecoEta_1 = np.array([])
    pfCandPt_1 = np.array([])
    pfCandPz_1 = np.array([])
    pfCandPtRel_1 = np.array([])
    pfCandPzRel_1 = np.array([])
    pfCandDr_1 = np.array([])
    pfCandDEta_1 = np.array([])
    pfCandDPhi_1 = np.array([])
    pfCandEta_1 = np.array([])
    pfCandDz_1 = np.array([])
    pfCandDzErr_1 = np.array([])
    pfCandDzSig_1 = np.array([])
    pfCandD0_1 = np.array([])
    pfCandD0Err_1 = np.array([])
    pfCandD0Sig_1 = np.array([])
    pfCandPuppiWeight_1 = np.array([])
    pfCandHits_1 = np.array([])
    pfCandPixHits_1 = np.array([])
    pfCandLostInnerHits_1 = np.array([])
    pfCandDVx_1 = np.array([])
    pfCandDVy_1 = np.array([])
    pfCandDVz_1 = np.array([])
    pfCandD_1 = np.array([])
    pfCandPdgid_1 = np.array([])
    pfCandCharge_1 = np.array([])
    pfCandFromPV_1 = np.array([])
    pfCandVtxQuality_1 = np.array([])
    pfCandTauIndMatch_1 = np.array([])
    pfCandHighPurityTrk_1 = np.array([])
    pfCandIsBarrel_1 = np.array([])
    lepHasSV_1 = np.array([])

    sig = bkg = 0
    time_per_event = 0
    count = 0

    for data in iter(test_loader):
        inputs, labels = data.x, data.y
        start = time.time()
        outputs = net(data)
        end = time.time()
        time_per_event += (end - start)/batch_size
        score = np.append(score,  outputs.detach().numpy())
        target = np.append(target, labels.detach().numpy()[:, 0])
        mva = np.append(mva, labels.detach().numpy()[:, 1])
        nLooseTaus = np.append(nLooseTaus, labels.detach().numpy()[:, 2])
        nPFCands_1 = np.append(nPFCands_1, labels.detach().numpy()[:, 3])
        decayMode_1 = np.append(decayMode_1, labels.detach().numpy()[:, 4])
        lepRecoPt_1 = np.append(lepRecoPt_1, labels.detach().numpy()[:, 5])
        lepRecoEta_1 = np.append(lepRecoEta_1, labels.detach().numpy()[:, 6])
        pfCandPt_1 = np.append(pfCandPt_1, labels.detach().numpy()[:, 7])
        pfCandPz_1 = np.append(pfCandPz_1, labels.detach().numpy()[:, 8])
        pfCandPtRel_1 = np.append(pfCandPtRel_1, labels.detach().numpy()[:, 9])
        pfCandPzRel_1 = np.append(pfCandPzRel_1, labels.detach().numpy()[:, 10])
        pfCandDr_1 = np.append(pfCandDr_1, labels.detach().numpy()[:, 11])
        pfCandDEta_1 = np.append(pfCandDEta_1, labels.detach().numpy()[:, 12])
        pfCandDPhi_1 = np.append(pfCandDPhi_1, labels.detach().numpy()[:, 13])
        pfCandEta_1 = np.append(pfCandEta_1, labels.detach().numpy()[:, 14])
        pfCandDz_1 = np.append(pfCandDz_1, labels.detach().numpy()[:, 15])
        pfCandDzErr_1 = np.append(pfCandDzErr_1, labels.detach().numpy()[:, 16])
        pfCandDzSig_1 = np.append(pfCandDzSig_1, labels.detach().numpy()[:, 17])
        pfCandD0_1 = np.append(pfCandD0_1, labels.detach().numpy()[:, 18])
        pfCandD0Err_1 = np.append(pfCandD0Err_1, labels.detach().numpy()[:, 19])
        pfCandD0Sig_1 = np.append(pfCandD0Sig_1, labels.detach().numpy()[:, 20])
        pfCandPuppiWeight_1 = np.append(pfCandPuppiWeight_1, labels.detach().numpy()[:, 21])
        pfCandHits_1 = np.append(pfCandHits_1, labels.detach().numpy()[:, 22])
        pfCandPixHits_1 = np.append(pfCandPixHits_1, labels.detach().numpy()[:, 23])
        pfCandLostInnerHits_1 = np.append(pfCandLostInnerHits_1, labels.detach().numpy()[:, 24])
        pfCandDVx_1 = np.append(pfCandDVx_1, labels.detach().numpy()[:, 25])
        pfCandDVy_1 = np.append(pfCandDVy_1, labels.detach().numpy()[:, 26])
        pfCandDVz_1 = np.append(pfCandDVz_1, labels.detach().numpy()[:, 27])
        pfCandD_1 = np.append(pfCandD_1, labels.detach().numpy()[:, 28])
        pfCandPdgid_1 = np.append(pfCandPdgid_1, labels.detach().numpy()[:, 29])
        pfCandCharge_1 = np.append(pfCandCharge_1, labels.detach().numpy()[:, 30])
        pfCandFromPV_1 = np.append(pfCandFromPV_1, labels.detach().numpy()[:, 31])
        pfCandVtxQuality_1 = np.append(pfCandVtxQuality_1, labels.detach().numpy()[:, 32])
        pfCandTauIndMatch_1 = np.append(pfCandTauIndMatch_1, labels.detach().numpy()[:, 33])
        pfCandHighPurityTrk_1 = np.append(pfCandHighPurityTrk_1, labels.detach().numpy()[:, 34])
        pfCandIsBarrel_1 = np.append(pfCandIsBarrel_1, labels.detach().numpy()[:, 35])
        lepHasSV_1 = np.append(lepHasSV_1, labels.detach().numpy()[:, 36])

        sig += target.sum()
        bkg += len(target) - target.sum()
        count += 1

    file = open("{0}evaluation.log".format(TRAINING_RES), 'w')
    file.write("Batch size: {0}; Average time per event: {1}; Signal: {2}; Background: {3}".format(batch_size, time_per_event/count, sig, bkg))
    file.close()

    df_eval = pd.DataFrame({"score": [i for i in score],
                            'label': [i for i in target],
                            'mva': [i for i in mva],
                            'nLooseTaus': [i for i in nLooseTaus],
                            'nPFCands_1': [i for i in nPFCands_1],
                            'decayMode_1': [i for i in decayMode_1],
                            'lepRecoPt_1': [i for i in lepRecoPt_1],
                            'lepRecoEta_1': [i for i in lepRecoEta_1],
                            'pfCandPt_1': [i for i in pfCandPt_1],
                            'pfCandPz_1': [i for i in pfCandPz_1],
                            'pfCandPtRel_1': [i for i in pfCandPtRel_1],
                            'pfCandPzRel_1': [i for i in pfCandPzRel_1],
                            'pfCandDr_1': [i for i in pfCandDr_1],
                            'pfCandDEta_1': [i for i in pfCandDEta_1],
                            'pfCandDPhi_1': [i for i in pfCandDPhi_1],
                            'pfCandEta_1': [i for i in pfCandEta_1],
                            'pfCandDz_1': [i for i in pfCandDz_1],
                            'pfCandDzErr_1': [i for i in pfCandDzErr_1],
                            'pfCandDzSig_1': [i for i in pfCandDzSig_1],
                            'pfCandD0_1': [i for i in pfCandD0_1],
                            'pfCandD0Err_1': [i for i in pfCandD0Err_1],
                            'pfCandD0Sig_1': [i for i in pfCandD0Sig_1],
                            'pfCandPuppiWeight_1': [i for i in pfCandPuppiWeight_1],
                            'pfCandHits_1': [i for i in pfCandHits_1],
                            'pfCandPixHits_1': [i for i in pfCandPixHits_1],
                            'pfCandLostInnerHits_1': [i for i in pfCandLostInnerHits_1],
                            'pfCandDVx_1': [i for i in pfCandDVx_1],
                            'pfCandDVy_1': [i for i in pfCandDVy_1],
                            'pfCandDVz_1': [i for i in pfCandDVz_1],
                            'pfCandD_1': [i for i in pfCandD_1],
                            'pfCandPdgid_1': [i for i in pfCandPdgid_1],
                            'pfCandCharge_1': [i for i in pfCandCharge_1],
                            'pfCandFromPV_1': [i for i in pfCandFromPV_1],
                            'pfCandVtxQuality_1': [i for i in pfCandVtxQuality_1],
                            'pfCandTauIndMatch_1': [i for i in pfCandTauIndMatch_1],
                            'pfCandHighPurityTrk_1': [i for i in pfCandHighPurityTrk_1],
                            'pfCandIsBarrel_1': [i for i in pfCandIsBarrel_1],
                            'lepHasSV_1': [i for i in lepHasSV_1]
                            })
    df_eval.to_csv("{1}EvalResults/ECN_{0}.csv".format(epoch, TRAINING_RES), index=False)
=======
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

>>>>>>> 1490b45245b585f86aad3cc9b868aacb6e7ace19
