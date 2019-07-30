import pandas as pd
import matplotlib.pyplot as plt
import sklearn.metrics as skmetrics


TRAIN_SET = "/nfs/dust/cms/user/dydukhle/TauIdSamples/TauId/2016/train_samples/"
TEST_SET = "/nfs/dust/cms/user/dydukhle/TauIdSamples/TauId/2016/test_samples/"
TRAINING_RES = "/nfs/dust/cms/user/bukinkir/TauId/GCN2/"
epoch = 10


def plot_output_dist(df, _dm=10):
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax.hist(df[(df['decay_mode'] == _dm)][(df['label'] == 1)]['mva'], bins=15, log=True, color='r',
            label='MVA True')
    ax.hist(df[(df['decay_mode'] == _dm)][(df['label'] == 0)]['mva'], bins=15, log=True, color='b',
            label='MVA Fakes')
    plt.title("Decay Mode: {0}".format(_dm))
    ax.legend()
    plt.xlabel("Output")
    plt.ylabel("Arn Units")

    ax2.hist(df[(df['decay_mode'] == _dm)][(df['label'] == 1)]['score'], bins=30, log=True, color='r',
             label="PF True")
    ax2.hist(df[(df['decay_mode'] == _dm)][(df['label'] == 0)]['score'], bins=30, log=True, color='b',
             label="PF Fakes")
    plt.title("Decay Mode: {0}".format(_dm))

    plt.legend()
    plt.xlabel("Output")
    plt.ylabel("Arn Units")
    plt.savefig('{0}/Out_dm{1}_{1}.pdf'.format(TRAINING_RES, _dm, epoch))
    return


if __name__ == '__main__':
    df = pd.read_csv("{1}EvalResults/GCN_{0}.csv".format(epoch, TRAINING_RES))

    fpr_dpf, tpr_dpf, _ = skmetrics.roc_curve(df.label, df.score)
    roc_auc_dpf = skmetrics.roc_auc_score(df.label, df.score)
    fpr_mva, tpr_mva, _ = skmetrics.roc_curve(df.label, df.mva)
    roc_auc_mva = skmetrics.roc_auc_score(df.label, df.mva)
    fpr_avg, tpr_avg, _ = skmetrics.roc_curve(df.label, (df.mva + df.score) / 2)
    roc_auc_avg = skmetrics.roc_auc_score(df.label, (df.mva + df.score) / 2)

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
    plt.savefig('{0}/ROC_CNN{1}.pdf'.format(TRAINING_RES, epoch))

    # Plot output distribution
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax.hist(df[(df['label'] == 1)]['mva'], bins=30, log=True, color='r', label='MVA True')
    ax.hist(df[(df['label'] == 0)]['mva'], bins=30, log=True, color='b', label='MVA Fakes')
    plt.xlabel("Output")
    plt.ylabel("Arn Units")
    ax.legend()
    ax2.hist(df[(df['label'] == 1)]['score'], bins=30, log=True, color='r', label="PF True")
    ax2.hist(df[(df['label'] == 0)]['score'], bins=30, log=True, color='b', label="PF Fakes")
    plt.xlabel("Output")
    plt.ylabel("Arn Units")
    plt.legend()
    plt.savefig('{0}/Out_dist{1}.pdf'.format(TRAINING_RES, epoch))

    # Plot output distributions for different decay modes
    for _dm in [0, 1, 10]:
        plot_output_dist(df, _dm)
