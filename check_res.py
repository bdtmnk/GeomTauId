from glob import glob
import pandas as pd


_file = "*.root*46*.csv"
MVA, LABELS, PRED = [],[],[]
files = glob(_file)
def fill(_file):

        df = pd.read_csv(_file)
        #print("KEYS:", df.keys())
        try:
                df = df[(df.decay_mode<=1)|(df.decay_mode==10)][(df.tau_match==1)]
                mva = df['mva']
                pred = df['valid_pred']
                label = df['labels_valid']
        except Exception:
                print(_file)
        else:
                MVA.append(mva)
                LABELS.append(label)
                PRED.append(pred)
        return

for _file in files:
        fill(_file)



df_mva = pd.concat(MVA)
df_mva.to_csv("MVA_46_update.csv")
df_pred = pd.concat(PRED)
df_pred.to_csv("PRED_46_update.csv")
df_labels = pd.concat(LABELS)
df_labels.to_csv("LABEL_46_update.csv")
