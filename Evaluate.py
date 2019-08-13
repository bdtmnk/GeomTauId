import numpy as np
import pandas as pd
import torch
from LoadData import TauIdDataset
from torch_geometric.data import  DataLoader
import time

TRAIN_SET = "/nfs/dust/cms/user/dydukhle/TauIdSamples/TauId/2016/train_samples/"
TEST_SET = "/nfs/dust/cms/user/dydukhle/TauIdSamples/TauId/2016/test_samples/"
TRAINING_RES = "/nfs/dust/cms/user/bukinkir/TauId/ECN3/"
epoch = 68
batch_size = 2048


def load_model(PATH):
    checkpoint = torch.load(PATH)
    net = checkpoint['net']
    optimizer = checkpoint['optimizer']
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    params_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print("Number of parameters: {0}".format(params_num))
    net.eval()
    return net, optimizer, epoch, loss


if __name__ == '__main__':
    test_dataset = TauIdDataset(TEST_SET, num=32768, mode='test')
    test_length = test_dataset.len
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=5)

    net, _, epoch, _ = load_model('{0}ECN_{1}.pt'.format(TRAINING_RES, epoch))

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
    pfCandPtRelPtRel_1 = np.array([])
    pfCandD0D0_1 = np.array([])
    pfCandDzDz_1 = np.array([])
    pfCandD0Dz_1 = np.array([])
    pfCandD0Dphi_1 = np.array([])
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
        pfCandPtRelPtRel_1 = np.append(pfCandPtRelPtRel_1, labels.detach().numpy()[:, 21])
        pfCandD0D0_1 = np.append(pfCandD0D0_1, labels.detach().numpy()[:, 22])
        pfCandDzDz_1 = np.append(pfCandDzDz_1, labels.detach().numpy()[:, 23])
        pfCandD0Dz_1 = np.append(pfCandD0Dz_1, labels.detach().numpy()[:, 24])
        pfCandD0Dphi_1 = np.append(pfCandD0Dphi_1, labels.detach().numpy()[:, 25])
        pfCandPuppiWeight_1 = np.append(pfCandPuppiWeight_1, labels.detach().numpy()[:, 26])
        pfCandHits_1 = np.append(pfCandHits_1, labels.detach().numpy()[:, 27])
        pfCandPixHits_1 = np.append(pfCandPixHits_1, labels.detach().numpy()[:, 28])
        pfCandLostInnerHits_1 = np.append(pfCandLostInnerHits_1, labels.detach().numpy()[:, 29])
        pfCandDVx_1 = np.append(pfCandDVx_1, labels.detach().numpy()[:, 30])
        pfCandDVy_1 = np.append(pfCandDVy_1, labels.detach().numpy()[:, 31])
        pfCandDVz_1 = np.append(pfCandDVz_1, labels.detach().numpy()[:, 32])
        pfCandD_1 = np.append(pfCandD_1, labels.detach().numpy()[:, 33])
        pfCandPdgid_1 = np.append(pfCandPdgid_1, labels.detach().numpy()[:, 34])
        pfCandCharge_1 = np.append(pfCandCharge_1, labels.detach().numpy()[:, 35])
        pfCandFromPV_1 = np.append(pfCandFromPV_1, labels.detach().numpy()[:, 36])
        pfCandVtxQuality_1 = np.append(pfCandVtxQuality_1, labels.detach().numpy()[:, 37])
        pfCandTauIndMatch_1 = np.append(pfCandTauIndMatch_1, labels.detach().numpy()[:, 38])
        pfCandHighPurityTrk_1 = np.append(pfCandHighPurityTrk_1, labels.detach().numpy()[:, 39])
        pfCandIsBarrel_1 = np.append(pfCandIsBarrel_1, labels.detach().numpy()[:, 40])
        lepHasSV_1 = np.append(lepHasSV_1, labels.detach().numpy()[:, 41])
        sig += target.sum()
        bkg += len(target) - target.sum()
        count += 1

    file = open("{0}eval.log".format(TRAINING_RES), 'w')
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
                            'pfCandPtRelPtRel_1': [i for i in pfCandPtRelPtRel_1],
                            'pfCandD0D0_1': [i for i in pfCandD0D0_1],
                            'pfCandDzDz_1': [i for i in pfCandDzDz_1],
                            'pfCandD0Dz_1': [i for i in pfCandD0Dz_1],
                            'pfCandD0Dphi_1': [i for i in pfCandD0Dphi_1],
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
                            'lepHasSV_1': [i for i in lepHasSV_1]})
    df_eval.to_csv("{1}EvalResults/ECN_{0}.csv".format(epoch, TRAINING_RES), index=False)
