import time
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import  accuracy_score, roc_auc_score,  precision_score, recall_score


class Logger():

    def __init__(self, training_res, model_name,  resume_training=False):
        if resume_training:
            self.log = open(training_res + "train.log", 'a')
        else:
            self.log = open(training_res + "train.log", 'w')
        self.loss = self.acc = self.prec = self.rec = self.auc = self.count = 0
        self.start_time = time.time()
        self.training_res = training_res
        self.model_name = model_name

    def eval_train(self, Loss, labels, outputs):

        # train_loss.append(np.array(loss.detach().numpy(), dtype="int32"))
        self.loss += Loss.item()
        # _, prediction = torch.max(outputs.data, 1)
        target = labels.detach().numpy().astype(int)
        score = outputs.detach().numpy()
        prediction = np.around(score)
        sig_num = target.sum()
        bkg_num = len(target) - sig_num
        print("{0} signal events; {1} background events".format(sig_num, bkg_num))
        self.acc += accuracy_score(target, prediction)
        self.prec += precision_score(target, prediction)
        self.rec += recall_score(target, prediction)
        self.auc += roc_auc_score(target, score)
        self.count += 1

    def save_train(self, epoch):
        epoch_time = time.time() - self.start_time
        try:
            self.log.write(
                "Epoch: {0}; Time: {1} s; Loss: {2}; Acc: {3}; AUC: {4}; Precision: {5}; Recall: {6}\n".format(
                    epoch, epoch_time, self.loss / self.count, self.acc / self.count, self.auc / self.count, self.prec / self.count, self.rec / self.count))
        except Exception, e:
            print(str(e))
        self.log.flush()
        self.loss = self.acc = self.prec = self.rec = self.auc = self.count = 0

    def eval_test(self, epoch,  Loss, labels, outputs):
        loss = 0
        score = np.array([])
        target = np.array([])
        # pt = np.array([])
        # eta = np.array([])
        # decay_mode = np.array([])
        # mva = np.array([])
        for i in range(len(Loss)):
            score = np.append(score, outputs[i].detach().numpy())
            target = np.append(target, labels[i].detach().numpy())
            # pt = np.append(pt, labels[i].detach().numpy()[:, 1])
            # eta = np.append(eta, labels[i].detach().numpy()[:, 2])
            # decay_mode = np.append(decay_mode, labels[i].detach().numpy()[:, 3])
            # mva = np.append(mva, labels[i].detach().numpy()[:, 4])
            loss += Loss[i].item()
        prediction = np.around(score)
        acc = accuracy_score(target, prediction)
        prec = precision_score(target, prediction)
        rec = recall_score(target, prediction)
        auc = roc_auc_score(target, score)
        try:
            self.log.write(
                "Test results({5} epoch):  Loss: {0}; Acc: {1}; AUC: {2}; Precision: {3}; Recall: {4}\n".format(loss / len(Loss), acc, auc, prec, rec, epoch))
        except Exception, e:
            print(str(e))
        self.log.flush()

        # df_eval = pd.DataFrame({"score": [i for i in score], 'label': [i for i in target], 'pt': [i for i in pt],
        #                     'eta': [i for i in eta], 'decay_mode': [i for i in decay_mode],
        #                     'mva': [i for i in mva]})
        # df_eval.to_csv("{1}EvalResults/{2}_{0}.csv".format(epoch, self.training_res, self.model_name), index=False)

    def close(self):
        self.log.close()