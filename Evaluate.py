import numpy as np
import pandas as pd
import torch
from LoadData import TauIdDataset
from torch_geometric.data import  DataLoader

TRAIN_SET = "/nfs/dust/cms/user/dydukhle/TauIdSamples/TauId/2016/train_samples/"
TEST_SET = "/nfs/dust/cms/user/dydukhle/TauIdSamples/TauId/2016/test_samples/"
TRAINING_RES = "/nfs/dust/cms/user/bukinkir/TauId/GCN2/"


def load_model(PATH):
    checkpoint = torch.load(PATH)
    net = checkpoint['net']
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    net.eval()
    return net, epoch, loss


if __name__ == '__main__':
    test_dataset = TauIdDataset(TEST_SET, mode='test')
    test_length = test_dataset.len
    test_loader = DataLoader(test_dataset, batch_size=test_length, shuffle=False, num_workers=0)

    net, epoch, _ = load_model('{0}GCN_{1}.pt'.format(TRAINING_RES, 50))

    score = np.array([])
    target = np.array([])
    pt = np.array([])
    eta = np.array([])
    decay_mode = np.array([])
    mva = np.array([])

    for i in range(10):
        for data in iter(test_loader):
            inputs, labels = data.x, data.y
            outputs = net(data)
            score =np. append(score,  np.exp(outputs.detach().numpy()[:, 1]))
            target = np.append(target, labels.detach().numpy()[:, 0])
            pt = np.append(pt, labels.detach().numpy()[:, 1])
            eta = np.append(eta, labels.detach().numpy()[:, 2])
            decay_mode = np.append(decay_mode, labels.detach().numpy()[:, 3])
            mva = np.append(mva, labels.detach().numpy()[:, 4])

    df_eval = pd.DataFrame(
            {"score": [i for i in score], 'label': [i for i in target], 'pt': [i for i in pt], 'eta': [i for i in eta],
             'decay_mode': [i for i in decay_mode], 'mva': [i for i in mva]})
    df_eval.to_csv("{1}EvalResults/GCN_{0}.csv".format(epoch, TRAINING_RES), index=False)
