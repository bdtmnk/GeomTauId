import time

import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score


class Logger:
    """Class for logs saving during training"""

    def __init__(self, training_res, model_name,  resume_training=False):
        """
        :param training_res: Directory where the logs will be stored
        :param model_name: Name of the model for which the logs are created
        :param resume_training: If the set  to False, the log file will be overwritten, otherwise logs will be appended to the existing file
        """
        if resume_training:
            self.log = open(training_res + "train.log", 'a')
        else:
            self.log = open(training_res + "train.log", 'w')
        self.loss = self.acc = self.prec = self.rec = self.auc = self.count = 0
        self.start_time = time.time()
        self.training_res = training_res
        self.model_name = model_name

    def eval_train(self, Loss, labels, outputs):
        """
        Evaluate the network performance (should be called after the each batch)

        :param Loss: Loss object
        :param labels: Targets for current batch
        :param outputs: Predictions for the current batch
        :return: None
        """

        self.loss += Loss.item()
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
        """
        Write the results of evaluation to the log file (should be called after each epoch)
        :param epoch: Number of epoch
        :return: None
        """
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
        for i in range(len(Loss)):
            score = np.append(score, outputs[i].detach().numpy())
            target = np.append(target, labels[i].detach().numpy())
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

    def close(self):
        """
        Close the log file (should be called after training end)

        :return: None
        """
        self.log.close()
