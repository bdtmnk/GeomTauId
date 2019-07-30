import pandas as pd
import matplotlib.pyplot as plt
import sklearn.metrics as skmetrics
from glob import glob
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


def plot_output_dist(df, _dm=10):
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax.hist(df[(df['decay_mode'] == _dm) & (df['labels_valid'] == 1)]['mva'], bins=15, log=True, color='r', histtype='step',
            label='MVA True')
    ax.hist(df[(df['decay_mode'] == _dm) & (df['labels_valid'] == 0)]['mva'], bins=15, log=True, color='b', histtype='step',
            label='MVA Fakes')
    plt.title("Decay Mode: {0}".format(_dm))
    ax.legend()
    plt.xlabel("Output")
    plt.ylabel("Arn Units")

    ax2.hist(df[(df['decay_mode'] == _dm) & (df['labels_valid'] == 1)]['valid_pred'], bins=30, log=True, color='r', histtype='step',
             label="DPF True")
    ax2.hist(df[(df['decay_mode'] == _dm) & (df['labels_valid'] == 0)]['valid_pred'], bins=30, log=True, color='b', histtype='step',
             label="DPF Fakes")
    plt.title("Decay Mode: {0}".format(_dm))

    plt.legend()
    plt.xlabel("Output")
    plt.ylabel("Arn Units")
    plt.savefig('{0}/Out_dm{1}_{2}.pdf'.format(config.get("model", "dir"), _dm,  config.get("model", "epoch")))
    return


if __name__ == '__main__':
    file_list = glob("{0}/ResultsClassFilter/*_{1}.csv".format(config.get("model", "dir"), config.get("model", "epoch")))
    df = pd.DataFrame()
    for file in file_list:
        print(file)
        df = df.append(pd.read_csv(file), ignore_index=True)
        # print(pd.read_csv(file).shape, df.shape)
    df.reindex()

    fpr_dpf, tpr_dpf, _ = skmetrics.roc_curve(df.labels_valid, df.valid_pred)
    roc_auc_dpf = skmetrics.roc_auc_score(df.labels_valid, df.valid_pred)
    fpr_mva, tpr_mva, _ = skmetrics.roc_curve(df.labels_valid, df.mva)
    roc_auc_mva = skmetrics.roc_auc_score(df.labels_valid, df.mva)
    fpr_avg, tpr_avg, _ = skmetrics.roc_curve(df.labels_valid, (df.mva + df.valid_pred) / 2)
    roc_auc_avg = skmetrics.roc_auc_score(df.labels_valid, (df.mva + df.valid_pred) / 2)

    # Plot inverted ROC curve
    plt.figure()
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

    # Plot output distribution
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax.hist(df[(df['labels_valid'] == 1)]['mva'], bins=30, log=True, color='r', label='MVA True', histtype='step')
    ax.hist(df[(df['labels_valid'] == 0)]['mva'], bins=30, log=True, color='b', label='MVA Fakes', histtype='step')
    plt.xlabel("Output")
    plt.ylabel("Arn Units")
    ax.legend()
    ax2.hist(df[(df['labels_valid'] == 1)]['valid_pred'], bins=30, log=True, color='r', label="DPF True", histtype='step')
    ax2.hist(df[(df['labels_valid'] == 0)]['valid_pred'], bins=30, log=True, color='b', label="DPF Fakes", histtype='step')
    plt.xlabel("Output")
    plt.ylabel("Arn Units")
    plt.legend()
    plt.savefig('{0}/Out_dist{1}.pdf'.format(config.get("model", "dir"), config.get("model", "epoch")))

    # Plot output distributions for different decay modes
    for _dm in [0, 1, 10]:
        plot_output_dist(df, _dm)