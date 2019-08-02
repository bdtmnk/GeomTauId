import pandas as pd
import numpy as np
from LoadData import TauIdDataset
import torch
from torch_geometric.data import Data, DataLoader
from LoadModel import Net
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import time
from sklearn.metrics import  accuracy_score, roc_auc_score,  precision_score, recall_score

TRAIN_SET = "/nfs/dust/cms/user/dydukhle/TauIdSamples/TauId/2016/train_samples/"
TEST_SET = "/nfs/dust/cms/user/dydukhle/TauIdSamples/TauId/2016/test_samples/"
TRAINING_RES = "/nfs/dust/cms/user/bukinkir/TauId/GCN2/"

if __name__ == "__main__":
    train_dataset = TauIdDataset(TRAIN_SET, num=4096)
    train_length = train_dataset.len
    print(train_length)
    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True, num_workers=5)

    test_dataset = TauIdDataset(TEST_SET, num=512)
    test_length = test_dataset.len
    test_loader = DataLoader(test_dataset, batch_size=test_length, shuffle=True, num_workers=5)

    net = Net()
    # optimizer = optim.Adam(net.parameters(), lr=0.001)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    running_loss = 0.0
    train_loss = []
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    # net = nn.DataParallel(net)

    i = 1

    log = open(TRAINING_RES + "train.log", 'w')
    start_time = time.time()

    for epoch in range(1, 101):
        Loss = acc = auc = prec = rec = 0
        j = k = 0
        for data in iter(train_loader):
            # print(data)
            inputs, labels = data.x, data.y
            print("Input shape", inputs.shape)
            print("Labels shape", labels.shape)

            # zero the parameter gradients
            optimizer.zero_grad()
            print("Num of features:",  data.num_features)

            # forward + backward + optimize
            outputs = net(data)
            print("Output shape", outputs.shape)
            loss = F.nll_loss(outputs, labels)
            loss.backward()
            # for par in net.parameters():
            #     print("Parameter:"+ str(par))
            #     print("Gradient:" + str(par.grad))
            optimizer.step()
            print(loss)

            train_loss.append(np.array(loss.detach().numpy(), dtype="int32"))
            Loss += loss.item()
            _, prediction= torch.max(outputs.data, 1)
            target =  labels.detach().numpy()
            score = np.exp(outputs.detach().numpy()[:, 1])
            acc += accuracy_score(target, prediction)
            prec += precision_score(target, prediction)
            rec += recall_score(target, prediction)
            i += 1
            j += 1
            try:
                auc += roc_auc_score(target, score)
                k += 1
            except Exception:
                  print("AUC is not defined\n")

        len = j
        epoch_time = time.time() - start_time
        try:
            log.write(
                "Epoch: {0}; Time: {1} s; Loss: {2}; Acc: {3}; AUC: {4}; Precision: {5}; Recall: {6}\n".format(
                    epoch, epoch_time, Loss / len, acc / len, auc / k, prec / len, rec / len))
        except Exception, e:
            print(str(e))
        log.flush()

        if epoch % 10 == 0:
            # test_dataiter = iter(test_loader)
            Loss = acc = auc = prec = rec = 0
            n = 0
            score = np.array([])
            target = np.array([])
            pt = np.array([])
            eta = np.array([])
            decay_mode = np.array([])
            mva = np.array([])
            prediction = np.array([])

            with torch.no_grad():
                for data in  iter(test_loader):
                    # inputs, labels = data.x, data.y
                    inputs, labels = data.x, data.y
                    outputs = net(data)
                    score = np.append(score, np.exp(outputs.detach().numpy()[:, 1]))
                    target = np.append(target, labels.detach().numpy())
                    # pt = np.append(pt, labels.detach().numpy()[:, 1])
                    # eta = np.append(eta, labels.detach().numpy()[:, 2])
                    # decay_mode = np.append(decay_mode, labels.detach().numpy()[:, 3])
                    # mva = np.append(mva, labels.detach().numpy()[:, 4])
                    loss = F.nll_loss(outputs, labels)
                    Loss += loss.item()
                    # prediction = np.array(outputs.detach().numpy()[:, 1] < outputs.detach().numpy()[:, 0], dtype='int32')
                    _, pred = torch.max(outputs.data, 1)
                    prediction = np.append(prediction, pred)
                    n += 1
            acc = accuracy_score(target, prediction)
            prec = precision_score(target, prediction)
            rec = recall_score(target, prediction)
            try:
                auc = roc_auc_score(target, score)
            except Exception:
                  print("AUC is not defined\n")
            try:
                log.write(
                "Test results: Epoch: {0}; Time: {1} s; Loss: {2}; Acc: {3}; AUC: {4}; Precision: {5}; Recall: {6}\n".format(
                    epoch, epoch_time, Loss/n, acc, auc, prec, rec))
            except Exception, e:
                print(str(e))
            log.flush()

            # df_eval = pd.DataFrame({"score": [i for i in score], 'label': [i for i in target], 'pt': [i for i in pt],
            #                         'eta': [i for i in eta], 'decay_mode': [i for i in decay_mode],
            #                         'mva': [i for i in mva]})
            # df_eval.to_csv("{1}EvalResults/GCN_{0}.csv".format(epoch, TRAINING_RES), index=False)

            df_eval = pd.DataFrame({"score": [i for i in score], 'label': [i for i in target]})
            df_eval.to_csv("{1}EvalResults/GCN_{0}.csv".format(epoch, TRAINING_RES), index=False)

            torch.save({
                'epoch': epoch,
                'net': net,
                'optimizer': optimizer,
                'loss': loss,
            }, '{0}GCN_{1}.pt'.format(TRAINING_RES, epoch))

    log.close()
    print('Finished Training:', running_loss)
