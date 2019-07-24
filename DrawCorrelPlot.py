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

if __name__ == "__main__":
    file_list = glob("{0}/ResultsClassFilter/*_{1}.csv".format(config.get("model", "dir"), config.get("model", "epoch")))
    df = pd.DataFrame()
    for file in file_list:
        print(file)
        df = df.append(pd.read_csv(file))
        print(pd.read_csv(file).shape, df.shape)
    df = df.sample(n=25000).reset_index(drop=True)
    df.sort_values('mva')
    plt.figure()
    plt.plot(df.valid_pred, df.mva, color='black', linestyle='', marker='o', label='Correlation plot', ms=0.25)
    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])
    plt.xlabel('DPF score')
    plt.ylabel('MVA score')
    # plt.legend(loc='upper left')
    plt.title('Correlation plot')
    plt.savefig('{0}/Correlation{1}.pdf'.format(config.get("model", "dir"), config.get("model", "epoch")))
