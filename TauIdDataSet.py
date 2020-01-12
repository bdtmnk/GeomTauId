""""
TauId DataSet:

""""

class TauIdDataset(Dataset):
    """
    Class for data loading from disk.
    """
    
    def __init__(self, root, mode='train', num = 1024, nfiles = 1, processes="all", scale=False):
        """
        :param root: Path to directory where ROOT files with data are stored
        :param mode: 'train' or 'test', in second case loads all the variables from tree with labels for evaluation
        :param num: Number of events to be used in one epoch
        :param nfiles: Number of files to load (the same number is used for signal and background files)
        """
        self.processes = processes
        self.root = root
        self.num = num
        self.mode = mode
        if self.mode == 'test':self.test_data = []
        
        self.sig_files = glob(root+"DY[1-4]*.root")
        self.lepton_files = glob(root+"DYJ*.root")
        
        if self.processes == "all":
            self.bkg_files =  glob(root+"QCD*.root") + glob(root+"W*.root")
        elif self.processes == "WJ":self.bkg_files = glob(root+"WJ*.root")
        elif self.processes == "QCD":self.bkg_files = glob(root+"QCD*.root")
       
        self.filenames =  self.bkg_files[:nfiles] + self.sig_files[:nfiles] + self.lepton_files[:nfiles]
        self.nfiles = 2*nfiles# len(self.filenames)

        np.random.shuffle(self.filenames)
        #TODO add shuffeling
        self.indices = []
        self.data = []
        self.jetdata = []
        self.mudata = []
        self.JETS = []
        self.cat_types = pd.Series([CategoricalDtype(categories=[1, 2, 11, 13, 130, 211, 22], ordered=True), 
                                    CategoricalDtype(categories=[-1, 0, 1], ordered=True),
                                    CategoricalDtype(categories=[0, 1], ordered=True),
                                    CategoricalDtype(categories=[0, 1, 5, 6, 10, 11], ordered=True),
                                    CategoricalDtype(categories=[-1, 1], ordered=True),

                                   ], index=CATEGORICAL_FEATURES)
        ##Divide batch size per n-files, n-event per file
        for i in range(len(self.filenames)):      
            LABELS = []
            GETX = True
            if i>self.nfiles:break
            #np.random.randint(1000)            #root_file.numentries
            try:
                root_file = uproot.open(self.filenames[i])['makeroottree']['AC1B']
            except Exception: continue

            tau_genmatch = root_file.pandas.df(['tau_genmatch'])

            if "DY" in self.filenames[i]:
                try:
                    tau_index = [tau_genmatch[tau_genmatch['tau_genmatch'] == 5].index[i][0] for i in range(num)]
                    tau_index+=tau_index
                except Exception:
                    tot_tau = tau_genmatch[tau_genmatch['tau_genmatch'] == 5].shape[0]
                    tau_index = [tau_genmatch[tau_genmatch['tau_genmatch'] == 5].index[i][0] for i in range(tot_tau)]
                finally:
                    if len(tau_index)>0: tau_index = list(np.random.choice(tau_index, num, replace=True))

                    
                try:
                    el_index =  [tau_genmatch[tau_genmatch['tau_genmatch'] == 1].index[i][0] for i in range(num)]
                    el_index += [tau_genmatch[tau_genmatch['tau_genmatch'] == 3].index[i][0] for i in range(num)]
                except Exception:
                    tot_el = tau_genmatch[tau_genmatch['tau_genmatch'] == 3].shape[0]
                    el_index = [tau_genmatch[tau_genmatch['tau_genmatch'] == 3].index[i][0] for i in range(tot_el)]
                finally:
                    if len(el_index)>0: el_index = list(np.random.choice(el_index, num, replace=True))

                try:
                    mu_index =  [tau_genmatch[tau_genmatch['tau_genmatch'] == 2].index[i][0] for i in range(num)]
                    mu_index += [tau_genmatch[tau_genmatch['tau_genmatch'] == 4].index[i][0] for i in range(num)]
                except Exception:
                    tot_mu = tau_genmatch[tau_genmatch['tau_genmatch'] == 4].shape[0]
                    mu_index = [tau_genmatch[tau_genmatch['tau_genmatch'] == 4].index[i][0] for i in range(tot_mu)]
                finally:
                    if len(mu_index)>0: mu_index = list(np.random.choice(mu_index, num, replace=True))
                
                #print("Tau:", tau_genmatch[tau_index])
                #print("El:", tau_genmatch[el_index])
                #print("Mu:", tau_genmatch[mu_index])
                indexes =  tau_index + el_index + mu_index
            else:
                
                try:
                    indexes =  [tau_genmatch[tau_genmatch['tau_genmatch'] == 6].index[i][0] for i in range(num)]
                except Exception:
                    tot_jet = tau_genmatch[tau_genmatch['tau_genmatch'] == 6].shape[0]
                    indexes =  [tau_genmatch[tau_genmatch['tau_genmatch'] == 6].index[i][0] for i in range(tot_jet)]
                #finally:
                #    indexes =  list(np.random.choice(indexes, num, replace=True))
                print("Jet:", len(indexes))
            print("Filename: ", self.filenames[i], ":", len(indexes))

            tracks = root_file.pandas.df(branches = PF_FEATURES[:],flatten=True)
            taus = root_file.pandas.df(branches = TAU_FEATURES[:],flatten=True)
            pfjets = root_file.pandas.df(branches = PF_JET_FEATURES[:], flatten=True)
            #eljets = root_file.pandas.df(branches = EL_FEATURE[:], flatten=True)
            #mujetx = root_file.pandas.df(branches = MU_FEATURE[:], flatten=True)
            index=0
            number_tau = 0
            particles = 0
            #while particles<tot_num:
            if "DY" in self.filenames[i]:tot_num = 3*num
            elif "QCD" in self.filenames[i]: tot_num = num
            while particles<tot_num: 
                for index in indexes:
                    try:
                        if root_file['tau_count'].array()[(index)]==0:continue
                    except Exception: print(e);break
                    #Get Labels and CrossValidate:
                    de = root_file["tau_byDeepTau2017v2p1VSeraw"].array()[index]
                    dj = root_file["tau_byDeepTau2017v2p1VSjetraw"].array()[index]
                    dmu = root_file["tau_byDeepTau2017v2p1VSmuraw"].array()[index]
                    LABELS = root_file['tau_genmatch'].array()[(index)]
                    MAP = {1:0, 
                           2:1,
                           3:0,
                           4:1,
                           5:2,
                           6:3}
                    if GETX:
                        jetmatchedTau = calc_dr(taus.loc[(index)], pfjets.loc[(index)], index, _type='pfjet')
                        try:
                            mumatchedTau = calc_dr(taus.loc[(index)], mujetx.loc[(index)], index, _type='muon')
                        except Exception as e:
                            mumatchedTau = []
                        try:
                            elmatchedTau = calc_dr(taus.loc[(index)], eljets.loc[(index)], index, _type='electron')
                        except Exception as e:
                            elmatchedTau = []

                        trackmatchedTau = calc_dr(taus.loc[(index)], tracks.loc[(index)], index)
                        for _tau in range(root_file['tau_count'].array()[(index)]):
                            if MAP[LABELS[_tau]] == 3 and "DY" in self.filenames[i]:continue
                            elif MAP[LABELS[_tau]] != 3 and "QCD" in self.filenames[i]:continue
                            if len(trackmatchedTau[_tau])==0: continue 
                            particles+=1

                            X_track = pd.DataFrame(tracks.loc[index].ix[trackmatchedTau[_tau]])
                            X_tau = pd.DataFrame(taus.loc[index].ix[_tau]).T
                            X_tau = pd.concat([X_tau for i in range(X_track.shape[0])])
                            X_tau = X_tau.reset_index()
                            X_track = X_track.reset_index()
                            X_track = X_track.join(X_tau)
                            X_track = X_track.astype(np.float32)

                            X_track['Y'] = MAP[LABELS[_tau]]

                            label = X_track['Y'][:1]
                            X_track['track_eta'] = (X_track['track_eta'] - X_track['tau_eta']).astype(np.float32)
                            X_track['track_phi'] = (X_track['track_phi'] - X_track['tau_phi']).astype(np.float32)
                            X_track['track_px'] = (X_track['track_pt']/X_track['tau_pt']).astype(np.float32)
                            X_track['track_py'] = (X_track['track_pz']/X_track['tau_pz']).astype(np.float32)
                            X_track['track_dz_sig'] = (X_track['track_dz']/X_track['track_dzerr']).astype(np.float32)
                            X_track['track_ID'] =    X_track['track_ID'].apply(lambda x: np.abs(x))
                            X_track['track_highPurity'] = X_track['track_highPurity'].astype(np.int32)

                            X_track['track_vx'] = X_track['track_vx'] - X_track['tau_vertexx']
                            X_track['track_vy'] = X_track['track_vy'] - X_track['tau_vertexy']
                            X_track['track_vz'] = X_track['track_vy'] - X_track['tau_vertexz']
                            X_track['track_D'] = np.sqrt(X_track['track_vx']**2 +  X_track['track_vy']**2 +  X_track['track_vz']**2)
                            X_track = pd.concat([X_track[np.concatenate((PF_VECTOR_FEATURES, ['Y']))],
                            pd.get_dummies(X_track[CATEGORICAL_FEATURES].apply(self.set_category, axis=0))], axis=1)
                            X_track['Y'] = MAP[LABELS[_tau]]

                            X_track = X_track.fillna(0)
                            self.data.append(X_track)  
            __l = [self.data[i]['Y'][0] for i in range(len(self.data))]
            df = pd.DataFrame({'Y':__l})
            #print("Len data:", len(self.data), particles)
        #np.random.choice(range(len(train_dataset.data)),size=2000,replace=True)

        self.len = len(self.data)
        #np.random.shuffle(self.data)
        #np.array([i.shape[0] for i in self.data]).sum()
        print("self.len:", self.len)
        
    @property
    def raw_file_names(self):
        return self.filenames


    @property
    def processed_file_names(self):
        return ['training.pt', 'test.pt']


    def __getitem__(self, index):
        """
        Get event with given index.
        :param index: Index of event to select
        :return: Pytorch tensor with features and labels for one event
        """
        np.random.seed(index)
        
        i = random.randint(0, len(self.data)-1 )
        if i > 0:
            #j = random.choice(self.indices[i])
            #self.indices[i] = np.delete(self.indices[i], np.where(self.indices[i] == j)[0])
            #try:
            return self.get_tensor(self.data[i])
            
        else:
            self.reload_file(i)
            i = self.nfiles - 1
            j = random.choice(self.indices[i])
            self.indices[i] = np.delete(self.indices[i], np.where(self.indices[i] == j)[0])
            return self.get_tensor(self.data[i].loc[j])


    def __len__(self):
        return self.len


    def reload_file(self, index):
        """
        Load new file instead of file with given index.
        Currently function isn't working properly
        :param index: Index of file to delete.
        :return: None
        """
        ##TODO REIMPLEMENT:
        filename = self.filenames[index]
        self.indices.remove(self.indices[index])
        self.data.remove(self.data[index])
        file = uproot.open(filename)
        i = self.nfiles - 1
        self.indices.append(get_indices(file['Candidates'], filename))
        if self.mode == 'train':
            self.data.append(file['Candidates'].pandas.df(
                np.concatenate((FEATURES, BINARY_FEATURES, CATEGORICAL_FEATURES, TARGET))).loc[self.indices[i]].astype(
                'float32'))
            self.data[i] = self.pre_process(self.data[i])
        elif self.mode == 'test':
            self.data.append(file['Candidates'].pandas.df().loc[self.indices[i]].astype('float32'))
            self.test_data.append(self.data[i])


    def set_category(self, value):
        """
        Apply one-hot encoding to categorical feature.
        :param value: Pandas series with categorical feature
        :return: Dataframe with encoded feature
        """
        return  value.astype(self.cat_types[value.name])


    def get_tensor(self, df, df_test=None):
        """
        Transform dataframe to pytorch tensor with features and labels.
        :param df: Dataframe to transform
        :param df_test: Dataframe with not pre-transformed features (only needed if mode='test')
        :return: Pytorch tensor
        """
        COORDINATES = ['track_eta','track_phi']
        label = df['Y'].iloc[:1]
        df = df.drop(columns='Y')
        
        pos = torch.tensor(df[COORDINATES].values)
        #df = df.drop(columns='track_count')
        x = torch.tensor(df.values)
        data = Data()
        data.pos = pos
        data.x = x
        
        data.y = torch.tensor(label.values,  dtype=torch.int64)
        return data