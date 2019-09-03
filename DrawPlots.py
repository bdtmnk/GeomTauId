from glob import glob

import matplotlib.pyplot as plt
import pandas as pd
import sklearn.metrics as skmetrics
from matplotlib import rcParams

rcParams.update({'figure.autolayout': True})

TRAIN_SET = "/nfs/dust/cms/user/dydukhle/TauIdSamples/TauId/2016/train_samples/"
TEST_SET = "/nfs/dust/cms/user/dydukhle/TauIdSamples/TauId/2016/test_samples/"
TRAINING_RES = "/nfs/dust/cms/user/bukinkir/TauId/ECN3/"
DPF_PATH = "/nfs/dust/cms/user/bukinkir/TauId/CNN_re/"
epoch = 68


def read_baseline(path, epoch):
    """
    Read evaluation results for DPF and MVA models

    :param path: Path to the folder where the evaluation results for DPF are stored
    :param epoch: Number of epoch for which to read results
    :return: Dataframe with evaluation results for DPF and MVA models
    """
    file_list = glob("{0}/ResultsClassFilter/*_{1}.csv".format(path, epoch))
    df = pd.DataFrame()
    for file in file_list:
        print(file)
        df = df.append(pd.read_csv(file), ignore_index=True)
    df.reindex()
    return df


def plot_pt_dist(df):
    """
    Draw a plot of the dependance of the output of the network and $p_T$

    :param df: Dataframe where the output and $p_T$  are stored
    :return: None
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    #ax1.scatter(df[(df['label'] == 1)]['mva'], df[(df['label'] == 1)]['lepRecoPt_1'], s = 0.01, color='r', label="MVA True")
    #ax1.scatter(df[(df['label'] == 0)]['mva'], df[(df['label'] == 0)]['lepRecoPt_1'], s = 0.01, color='b', label="MVA Fakes")
    #plt.xlabel("Output")
    #plt.ylabel("p_T")
    #ax1.legend()

    #ax2.scatter(dpf[(dpf['labels_valid'] == 1)]['lepRecoPt_1'], ((dpf[(dpf['labels_valid'] == 1)]['mva'] + 1) / 2 + dpf[(dpf['labels_valid'] == 1)]['valid_pred']) / 2,
    #            color='r', label='DPF True')
    #ax2.scatter(dpf[(dpf['labels_valid'] == 0)]['lepRecoPt_1'],
    #            ((dpf[(dpf['labels_valid'] == 0)]['mva'] + 1) / 2 + dpf[(dpf['labels_valid'] == 0)]['valid_pred']) / 2,
    #            color='b', label='DPF Fakes')
    #ax2.legend()
    #plt.xlabel("Output")
    #plt.ylabel("p_T")

    ax1.scatter(df[(df['label'] == 1)]['score'], df[(df['label'] == 1)]['lepRecoPt_1'], s = 0.02, color='r', label="ECN True")
    ax2.scatter(df[(df['label'] == 0)]['score'], df[(df['label'] == 0)]['lepRecoPt_1'], s = 0.02, color='b', label="ECN Fakes")
    #ax1.legend()
    #ax2.legend()
    ax1.set_xlabel("Output")
    ax1.set_ylabel("$p_T$")
    ax1.set_title("True tau")
    ax2.set_xlabel("Output")
    ax2.set_ylabel("$p_T$")
    ax2.set_title("Fake tau")
    plt.savefig('{0}/pt_dist{1}.pdf'.format(TRAINING_RES, epoch))


def plot_output_dist(df, dpf,  _dm=10):
    """
    Plot the output distribution for the given decay mode

    :param df: Dataframe with the evaluation results for ECN
    :param dpf: Dataframe with the evaluation results for DPF and MVA
    :param _dm: Number of decay mode
    :return: None
    """
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 3.5))

    roc_auc_mva = skmetrics.roc_auc_score(df[(df['decayMode_1'] == _dm)]['label'], df[(df['decayMode_1'] == _dm)]['mva'])
    roc_auc_dpf = skmetrics.roc_auc_score(dpf[(dpf['decay_mode'] == _dm)]['labels_valid'], ((dpf[(dpf['decay_mode'] == _dm)]['mva'] + 1) / 2 + dpf[(dpf['decay_mode'] == _dm)]['valid_pred']) / 2)
    roc_auc_gcn = skmetrics.roc_auc_score(df[(df['decayMode_1'] == _dm)]['label'], df[(df['decayMode_1'] == _dm)]['score'])

    ax1.hist(df[(df['decayMode_1'] == _dm)][(df['label'] == 1)]['mva'], bins=30, log=True, color='r', label="Tau", histtype='step')
    ax1.hist(df[(df['decayMode_1'] == _dm)][(df['label'] == 0)]['mva'], bins=30, log=True, color='b', label="Jets", histtype='step')
    ax1.set_xlabel("Output")
    ax1.set_ylabel("Number of events")
    ax1.set_title("MVA (ROC AUC = {:.3f})".format(roc_auc_mva))
    ax1.legend(loc='lower center')

    ax2.hist(((dpf[(dpf['decay_mode'] == _dm)][(dpf['labels_valid'] == 1)]['mva'] + 1) / 2 + dpf[(dpf['decay_mode'] == _dm)][(dpf['labels_valid'] == 1)]['valid_pred']) / 2, bins=30, log=True,
            color='r', label='Tau', histtype='step')
    ax2.hist(((dpf[(dpf['decay_mode'] == _dm)][(dpf['labels_valid'] == 0)]['mva'] + 1) / 2 + dpf[(dpf['decay_mode'] == _dm)][(dpf['labels_valid'] == 0)]['valid_pred']) / 2, bins=30, log=True,
            color='b', label='Jets', histtype='step')
    # ax.hist((dpf[(dpf['decayMode_1'] == _dm)][(dpf['labels_valid'] == 1)]['mva'] +
    #          dpf[(dpf['decayMode_1'] == _dm)][(dpf['labels_valid'] == 1)]['valid_pred']) / 2, bins=30, log=True,
    #         color='r', label='DPF True', histtype='step')
    # ax.hist((dpf[(dpf['decayMode_1'] == _dm)][(dpf['labels_valid'] == 0)]['mva'] +
    #          dpf[(dpf['decayMode_1'] == _dm)][(dpf['labels_valid'] == 0)]['valid_pred']) / 2, bins=30, log=True,
    #         color='b', label='DPF Fakes', histtype='step')
    # plt.title("Decay Mode: {0}".format(_dm))
    ax2.legend(loc='lower center')
    ax2.set_xlabel("Output")
    ax2.set_title("DPF + MVA (ROC AUC = {:.3f})".format(roc_auc_dpf))

    ax3.hist(df[(df['decayMode_1'] == _dm)][(df['label'] == 1)]['score'], bins=30, log=True, color='r',
             label="Tau", histtype='step')
    ax3.hist(df[(df['decayMode_1'] == _dm)][(df['label'] == 0)]['score'], bins=30, log=True, color='b',
             label="Jets", histtype='step')
    # plt.title("Decay Mode: {0}".format(_dm))

    ax3.legend(loc='lower center')
    ax3.set_xlabel("Output")
    ax3.set_title("ECN (ROC AUC = {:.3f})".format(roc_auc_gcn))
    plt.savefig('{0}/Out_dm{1}_{1}.pdf'.format(TRAINING_RES, _dm, epoch))
    return


if __name__ == '__main__':
    df = pd.read_csv("{1}EvalResults/ECN_{0}.csv".format(epoch, TRAINING_RES))
    dpf = read_baseline(DPF_PATH, 50)

    # plt.tight_layout()

    # df = df[(df['lepRecoPt_1']>50) & (df['lepRecoPt_1']<70) & (abs(df['lepRecoEta_1']) < 2.3) & (abs(df['pfCandDz_1']) < 0.2)]

    # fpr_gcn, tpr_gcn, _ = skmetrics.roc_curve(df.label, df.score)
    # roc_auc_gcn = skmetrics.roc_auc_score(df.label, df.score)
    # fpr_mva, tpr_mva, _ = skmetrics.roc_curve(df.label, df.mva)
    # roc_auc_mva = skmetrics.roc_auc_score(df.label, df.mva)

    fpr_mva, tpr_mva, _ = skmetrics.roc_curve(dpf.labels_valid, dpf.mva)
    roc_auc_mva = skmetrics.roc_auc_score(dpf.labels_valid, dpf.mva)
    fpr_dpf, tpr_dpf, _ = skmetrics.roc_curve(dpf.labels_valid, ((dpf.mva + 1)/2  + dpf.valid_pred) / 2)
    roc_auc_dpf = skmetrics.roc_auc_score(dpf.labels_valid, ((dpf.mva + 1)/2 + dpf.valid_pred) / 2)
    # fpr_dpf0, tpr_dpf0, _ = skmetrics.roc_curve(dpf.labels_valid, dpf.valid_pred)
    # roc_auc_dpf0 = skmetrics.roc_auc_score(dpf.labels_valid, dpf.valid_pred)
    fpr_gcn, tpr_gcn, _ = skmetrics.roc_curve(df.label, df.score)
    roc_auc_gcn = skmetrics.roc_auc_score(df.label, df.score)
    # fpr_mva, tpr_mva, _ = skmetrics.roc_curve(df[(df['lepRecoPt_1']>50) & (df['lepRecoPt_1']<70)].label, df[(df['lepRecoPt_1']>50) & (df['lepRecoPt_1']<70)].mva)
    # roc_auc_mva = skmetrics.roc_auc_score(df[(df['lepRecoPt_1']>50) & (df['lepRecoPt_1']<70)].label, df[(df['lepRecoPt_1']>50) & (df['lepRecoPt_1']<70)].mva)
    # fpr_dpf, tpr_dpf, _ = skmetrics.roc_curve(dpf.labels_valid, ((dpf.mva + 1) / 2 + dpf.valid_pred) / 2)
    # roc_auc_dpf = skmetrics.roc_auc_score(dpf.labels_valid, ((dpf.mva + 1) / 2 + dpf.valid_pred) / 2)
    # fpr_dpf, tpr_dpf, _ = skmetrics.roc_curve(dpf.labels_valid, (dpf.mva + dpf.valid_pred) / 2)
    # roc_auc_dpf = skmetrics.roc_auc_score(dpf.labels_valid, (dpf.mva + dpf.valid_pred) / 2)

    # Plot inverted ROC curve
    plt.figure()
    plt.plot(tpr_gcn, fpr_gcn, color='red', label='ECN: ROC AUC={:.3f}'.format(roc_auc_gcn))
    # plt.plot(tpr_dpf0, fpr_dpf0, color='red', label='DPF: ROC AUC={:.3f}'.format(roc_auc_dpf0))
    plt.plot(tpr_dpf, fpr_dpf, color='blue', label='DPF + MVA: ROC AUC={:.3f}'.format(roc_auc_dpf))
    plt.plot(tpr_mva, fpr_mva, color='green', label='MVA: ROC AUC={:.3f}'.format(roc_auc_mva))
    plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    # plt.yscale('log')
    plt.xlabel('True Positive')
    plt.ylabel('False Positive')
    plt.legend(loc='upper left')
    plt.title('Inverted ROC\n')
    plt.savefig('{0}/ROC_ECN{1}.pdf'.format(TRAINING_RES, epoch))

    # Plot output distribution
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 3.5))
    ax1.hist(df[(df['label'] == 1)]['mva'], bins=30, log=True, color='r', label="Tau", histtype='step')
    ax1.hist(df[(df['label'] == 0)]['mva'], bins=30, log=True, color='b', label="Jets", histtype='step')
    ax1.set_xlabel("Output")
    ax1.set_ylabel("Number of events")
    ax1.set_title("MVA (ROC AUC = {:.3f})".format(roc_auc_mva))
    ax1.legend(loc='lower center')
    ax2.hist(((dpf[(dpf['labels_valid'] == 1)]['mva'] + 1)/2 + dpf[(dpf['labels_valid'] == 1)]['valid_pred'])/2, bins=30, log=True, color='r', label='Tau', histtype='step')
    ax2.hist(((dpf[(dpf['labels_valid'] == 0)]['mva'] + 1)/2 + dpf[(dpf['labels_valid'] == 0)]['valid_pred'])/2, bins=30, log=True, color='b', label='Jets', histtype='step')
    ax2.set_xlabel("Output")
    ax2.set_title("DPF + MVA (ROC AUC = {:.3f})".format(roc_auc_dpf))
    ax2.legend(loc='lower center')
    ax3.hist(df[(df['label'] == 1)]['score'], bins=30, log=True, color='r', label="Tau", histtype='step')
    ax3.hist(df[(df['label'] == 0)]['score'], bins=30, log=True, color='b', label="Jets", histtype='step')
    ax3.set_xlabel("Output")
    ax3.set_title("ECN (ROC AUC = {:.3f})".format(roc_auc_gcn))
    ax3.legend(loc='lower center')
    plt.savefig('{0}/Out_dist_plot{1}.pdf'.format(TRAINING_RES, epoch))

    # plot_pt_dist(df)

    # Plot output distributions for different decay modes
    for _dm in [0, 1, 10]:
        plot_output_dist(df, dpf, _dm)
