from glob import glob
import sklearn.metrics as skmetrics
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import ConfigParser

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--config", default="config.ini", help="Configuration file")
args = parser.parse_args()
configuration_name = args.config
print(configuration_name)

# Parse config
config = ConfigParser.RawConfigParser()
config.read(configuration_name)

# def print_roc(label, pred, path ):
#
#     fpr, tpr, thresholds = skmetrics.roc_curve(label, pred)
#     #print(fpr[-5:], tpr[-5:])
#     #print(fpr[:5], tpr[:5])
#     print(skmetrics.roc_auc_score(label, pred))
#     plt.figure()
#     plt.plot(tpr, fpr, color='red', label='ROC')
#     plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')
#     plt.xlim([0.0, 1.05])
#     plt.ylim([0.0, 1.05])
#     plt.xlabel('True Positive')
#     plt.ylabel('False Positive')
#     plt.title('Inverted ROC')
#     plt.savefig(path + 'ROC_CNN{0}.pdf'.format(config.get("model", "epoch")))
#     return

if __name__ == "__main__":
    file_list = glob("{0}/ResultsClassFilter/*_{1}.csv".format(config.get("model", "dir"), config.get("model", "epoch")))
    df = pd.DataFrame()
    for file in file_list:
        print(file)
        df = df.append(pd.read_csv(file))
        print(pd.read_csv(file).shape, df.shape)
    #df.sort_values('valid_pred')
    #print(df.labels_valid[:10], df.valid_pred[:10])
    #print(df.labels_valid[-10:], df.valid_pred[-10:])
    #print_roc(df.tau_match, df.valid_pred, "{0}/".format( config.get("model", "dir")))
    fpr_dpf, tpr_dpf, _ = skmetrics.roc_curve(df.labels_valid, df.valid_pred)
    roc_auc_dpf = skmetrics.roc_auc_score(df.labels_valid, df.valid_pred)
    fpr_mva, tpr_mva, _ = skmetrics.roc_curve(df.labels_valid, df.mva)
    roc_auc_mva = skmetrics.roc_auc_score(df.labels_valid, df.mva)
    fpr_avg, tpr_avg, _ = skmetrics.roc_curve(df.labels_valid, (df.mva + df.valid_pred)/2)
    roc_auc_avg= skmetrics.roc_auc_score(df.labels_valid, (df.mva + df.valid_pred)/2)
    #print(fpr[-5:], tpr[-5:])
    #print(fpr[:5], tpr[:5])
    plt.figure()
    print(len(tpr_dpf))
    plt.plot(tpr_dpf, fpr_dpf, color='red', label='DPF:ROC AUC={:.3f}'.format(roc_auc_dpf))
    plt.plot(tpr_mva, fpr_mva, color='blue', label='MVA:ROC AUC={:.3f}'.format(roc_auc_mva))
    plt.plot(tpr_avg, fpr_avg, color='green', label='Average:ROC AUC={:.3f}'.format(roc_auc_avg))
    plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')
    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])
    plt.xlabel('True Positive')
    plt.ylabel('False Positive')
    plt.legend(loc='upper left')
    plt.title('Inverted ROC')
    plt.savefig('{0}/ROC_CNN{1}.pdf'.format(config.get("model", "dir"), config.get("model", "epoch")))
