#!/usr/bin/python
# Written by Owen Colegrove
# This class is responsible for overseeing training batch preparation
# For a better understanding of the code structure, refer to "Trainer.py"
from glob import glob
import ROOT as r
import numpy as np
import root_numpy as rn
# from multiprocessing import Pool
from multiprocessing.pool import ThreadPool
from random import randint
from EventFiller import EventFiller
import pandas as pd


class Utilities():

    def __init__(self, maxCands):
        self.maxCands = maxCands

    # Build Event Weights For Training (flatten in observed tau pt and do class-balancing)
    def GetEventWeights(self, pt_train, Y):
        W_train = np.zeros(len(Y)) + 1
        bins_by_pt = np.append(np.arange(30, 100, 10), [10000])

        ptBkg = pt_train[Y != 1]
        Hbkg, xedges = np.histogram(ptBkg, bins_by_pt)
        Hbkg = 1. / Hbkg
        Hbkg[np.isnan(Hbkg)] = 1.
        Hbkg[np.isinf(Hbkg)] = 1.
        ptSig = pt_train[Y == 1]
        Hsig, xedges = np.histogram(ptSig, bins_by_pt)
        Hsig = 1. / Hsig
        Hsig[np.isnan(Hsig)] = 1.
        Hsig[np.isinf(Hsig)] = 1.
        for id_ in range(len(pt_train)):
            ptInd = np.searchsorted(xedges, pt_train[id_]) - 1
            if (Y[id_] == 1):
                W_train[id_] = Hsig[ptInd] * W_train[id_]
            else:
                W_train[id_] = Hbkg[ptInd] * W_train[id_]

        W_train[Y == 0] = W_train[Y == 0] * len(W_train) * .5 / np.sum(W_train[Y == 0])
        W_train[Y == 1] = W_train[Y == 1] * len(W_train) * .5 / np.sum(W_train[Y == 1])

        # print "Printing max,min,mean of signal weights"
        # print np.max(W_train[Y == 0]),np.min(W_train[Y == 0]),np.mean(W_train[Y == 0])
        return W_train

    # Convert ROOT file to python array via the EventFiller class
    def RootToArr(self, inh5, X_pfs, X_mva, Y, Z, MVA):
        # Specify the maximum number of particle flow candidates and the number of features to be loaded
        for event in xrange(len(inh5)):
            # Load an Event
            levent = inh5.iloc[event]
            filler = EventFiller(levent)

            # Fill the event array up to the maxium number of particle candidates
            evt = filler.Fill(self.maxCands)
            try:
                zeros = np.zeros((self.maxCands - len(evt), evt.shape[1]))
                evt = np.vstack((zeros, evt))
            except Exception:
                continue
            # print "evt shape", evt.shape
            X_pfs.append(evt)
            Y.append(int(levent['lepTauMatch_1']))
            Z.append(int(levent['lepRecoPt_1']))
	   
            MVA.append(np.array([levent['lepMVAIso_1'], int(levent['decayMode_1']), int(levent['lepMuMatch_1']),
                                 int(levent['lepEleMatch_1']), levent['lepTauMatch_1'], 1-levent['lepTauMatch_1']]))
            #print("MVA shape:", MVA.shape)
        return np.array(X_pfs), np.array(Y), np.array(Z), np.array(MVA)

    # Load a specific root file into a processed array
    def LoadFile(self, in_file, start, stop):
        X_pfs, X_mva, Y, Z, MVA = [], [], [], [], []
        f = r.TFile.Open(in_file)
        try:
            t = f.Get("Candidates")
        except:
            return [], [], [], [], []
        #arr = pd.DataFrame(rn.tree2array(t, start=start, stop=stop))
        arr = pd.DataFrame(rn.tree2array(t))
        if "DY" in in_file:
            arr = arr[((arr['decayMode_1']<=1) | (arr['decayMode_1']==10)) & (arr['lepTauMatch_1'] == 1) ]
        elif "WJ" in in_file:
            arr = arr[((arr['decayMode_1']<=1) | (arr['decayMode_1']==10)) & (arr['lepTauMatch_1'] == 0) ]
        if stop <= len(arr.index):
            arr = arr[start:stop]
        else:
            arr = arr[start:]
    #arr = arr[(arr['decayMode_1']<=1) | (arr['decayMode_1']==10)]
    #print(arr.head(2))
        #print("df.shape: ", arr.shape)
        #try:
        X_pfs, Y, Z, MVA = self.RootToArr(arr, X_pfs, X_mva, Y, Z, MVA)
        #except Exception:
        #    print(in_file)
        #    print(in_file)

        return np.array(X_pfs), np.array(Y), np.array(Z), np.array(MVA)

    def BuildValidationDataset(self,file, nEvents=None):
        """

        :return:
        """
        X_1, Y, Z, MVA = [], [], [], []
        #try:
        f = r.TFile.Open(file)
        t = f.Get("Candidates")
        start = 0#randint(0, t.GetEntries() - nEvents)
        if nEvents is None:
            stop = start+t.GetEntries()
        else:
            stop = start+nEvents
        print(start, stop)
        if (len(X_1) == 0):
            X_1, Y, Z, MVA = self.LoadFile(file, start, stop)
        else:
            X_1t, Yt, Zt, MVAt = self.LoadFile(file, start, stop)
            X_1 = np.vstack((X_1, X_1t))
            Y = np.append(Y, Yt)
            Z = np.append(Z, Zt)
            MVA = np.vstack((MVA, MVAt))
        #except Exception:
        #    print("File: ", file)
        #    return np.array(X_1), np.array(Y), np.array(Z), np.array(MVA)
        return np.array(X_1), np.array(Y), np.array(Z), np.array(MVA)

    # Build a dataset from a list of files, loading each into an array and appending the results
    def BuildDataset(self, list_of_files, nEvents=5000):
        X_1, Y, Z, MVA = [], [], [], []
        for infile in list_of_files:
            try:
                f = r.TFile.Open(infile)
                t = f.Get("Candidates")
                #arr = pd.DataFrame(rn.tree2array(t))
            	#if "DY" in infile:
                #    arr = arr[((arr['decayMode_1']<=1) | (arr['decayMode_1']==10)) & (arr['lepTauMatch_1'] == 1) ]
       		#elif "WJ" in infile:
            	#    arr = arr[((arr['decayMode_1']<=1) | (arr['decayMode_1']==10)) & (arr['lepTauMatch_1'] == 0) ]
                start = 0#randint(0, len(arr.index) - nEvents)
                stop = start + nEvents
                if (len(X_1) == 0):
                    X_1, Y, Z, MVA = self.LoadFile(infile, start, stop)
                else:
                    X_1t, Yt, Zt, MVAt = self.LoadFile(infile, start, stop)
                    X_1 = np.vstack((X_1, X_1t))
                    Y = np.append(Y, Yt)
                    Z = np.append(Z, Zt)
                    MVA = np.vstack((MVA, MVAt))
            except Exception:
                print("File: ", infile)
                continue
        return np.array(X_1), np.array(Y), np.array(Z), np.array(MVA)

    # Build a training batch with a taget directory, using specified number of processes and files.
    def BuildBatch(self, indir='/beegfs/desy/user/dydukhle/TauId/new_train_samples/*.root', nProcs=5, nFiles=10, nEvents=5000):
        file_list = glob(indir)
        np.random.shuffle(file_list)
        pool = ThreadPool(processes=nProcs)
        poolblock = []
        for p in np.arange(0, nProcs * nFiles, nFiles):
            for f in file_list[p:p + nFiles]:
                print(f.split("/")[-1])
            poolblock.append(pool.apply_async(self.BuildDataset, (file_list[p:p + nFiles],nEvents)))
            X_1, Y, Z, MVA = [], [], [], []
        for res in poolblock:
            try:
                if (len(X_1) == 0):
                    output = res.get(timeout=2000)
                    X_1, Y, Z, MVA = output  # res.get(timeout=1000)
                else:
                    X_1t, Yt, Zt, MVAt = res.get(timeout=1000)
                    if (len(X_1t) > 0):
                        X_1 = np.vstack((X_1, X_1t))
                        Y = np.append(Y, Yt)
                        Z = np.append(Z, Zt)
                        MVA = np.vstack((MVA, MVAt))
            except:
                #print "Error"
                continue

        pool.close()
        return X_1, Y, self.GetEventWeights(Z, Y), MVA

    def BuildValid(self, DY_X_1, DY_Y_1, WJ_X_1, WJ_Y_1, MVA_DY, MVA_WJ):

        DY_X_1 = DY_X_1[DY_Y_1 == 1]
        DY_Y_1 = DY_Y_1[DY_Y_1 == 1]
        MVA_DY = MVA_DY[DY_Y_1 == 1]

        WJ_X_1 = WJ_X_1[WJ_Y_1 == 0]
        WJ_Y_1 = WJ_Y_1[WJ_Y_1 == 0]
        MVA_WJ = MVA_WJ[WJ_Y_1 == 0]

        VALID_X_1 = np.vstack((DY_X_1, WJ_X_1))
        VALID_Y = np.append(DY_Y_1, WJ_Y_1)
        MVA = np.vstack((MVA_DY, MVA_WJ))
        return VALID_X_1, VALID_Y, MVA

