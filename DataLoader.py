import os
import uproot
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


"""
Define the path and the global varibales here:
"""
PATH = "/pnfs/desy.de/cms/tier2/store/user/ldidukh/TauId/2016/train_samples/"
DY_LIST = os.listdir(PATH_2018+PROCESS)
root_file = uproot.open(PATH_2018+PROCESS+"/DY1JetsToLL_M-50_3226.root")['makeroottree/AC1B']

PI =3.14

PF_FEATURES = ['track_count',
 'track_px',
 'track_py',
 'track_pz',
 'track_pt',
 'track_eta',
 'track_phi',
 'track_charge',
 'track_mass',
 'track_dxy',
 'track_dxyerr',
 'track_dz',
 'track_dzerr',
 'track_vx',
 'track_vy',
 'track_vz',
 'track_ID',
 'track_highPurity']

TAU_FEATURES = [
 'tau_count',
 'tau_helixparameters',
 'tau_helixparameters_covar',
 'tau_referencePoint',
 'tau_Bfield',
 'tau_e',
 'tau_px',
 'tau_py',
 'tau_pz',
 'tau_mass',
 'tau_eta',
 'tau_phi',
 'tau_pt',
 'tau_vertexx',
 'tau_vertexy',
 'tau_vertexz',
 'tau_decayMode'
 'tau_dxy',
 'tau_dxySig',
 'tau_dz',
 'tau_ip3d',
 'tau_ip3dSig',
 'tau_charge',
 'tau_flightLength',
 'tau_flightLengthSig',
 'tau_chargedIsoPtSum',
 'tau_chargedIsoPtSumdR03',
 'tau_decayModeFinding',
 'tau_decayModeFindingNewDMs',
 'tau_footprintCorrection',
 'tau_footprintCorrectiondR03',
 'tau_neutralIsoPtSum',
 'tau_neutralIsoPtSumWeight',
 'tau_neutralIsoPtSumWeightdR03',
 'tau_neutralIsoPtSumdR03',
 'tau_photonPtSumOutsideSignalCone',
 'tau_photonPtSumOutsideSignalConedR03'
]

TAGGERS = [
 'tau_againstElectronLooseMVA6',
 'tau_againstElectronMVA6Raw',
 'tau_againstElectronMVA6category',
 'tau_againstElectronMediumMVA6',
 'tau_againstElectronTightMVA6',
 'tau_againstElectronVLooseMVA6',
 'tau_againstElectronVTightMVA6',
 'tau_againstMuonLoose3',
 'tau_againstMuonTight3',
 'tau_byCombinedIsolationDeltaBetaCorrRaw3Hits',
 'tau_byIsolationMVArun2v1DBdR03oldDMwLTraw',
 'tau_byIsolationMVArun2v1DBnewDMwLTraw',
 'tau_byIsolationMVArun2v1DBoldDMwLTraw',
 'tau_byIsolationMVArun2v1PWdR03oldDMwLTraw',
 'tau_byIsolationMVArun2v1PWnewDMwLTraw',
 'tau_byIsolationMVArun2v1PWoldDMwLTraw',
 'tau_byLooseCombinedIsolationDeltaBetaCorr3Hits',
 'tau_byLooseIsolationMVArun2v1DBdR03oldDMwLT',
 'tau_byLooseIsolationMVArun2v1DBnewDMwLT',
 'tau_byLooseIsolationMVArun2v1DBoldDMwLT',
 'tau_byLooseIsolationMVArun2v1PWdR03oldDMwLT',
 'tau_byLooseIsolationMVArun2v1PWnewDMwLT',
 'tau_byLooseIsolationMVArun2v1PWoldDMwLT',
 'tau_byMediumCombinedIsolationDeltaBetaCorr3Hits',
 'tau_byMediumIsolationMVArun2v1DBdR03oldDMwLT',
 'tau_byMediumIsolationMVArun2v1DBnewDMwLT',
 'tau_byMediumIsolationMVArun2v1DBoldDMwLT',
 'tau_byMediumIsolationMVArun2v1PWdR03oldDMwLT',
 'tau_byMediumIsolationMVArun2v1PWnewDMwLT',
 'tau_byMediumIsolationMVArun2v1PWoldDMwLT',
 'tau_byPhotonPtSumOutsideSignalCone',
 'tau_byTightCombinedIsolationDeltaBetaCorr3Hits',
 'tau_byTightIsolationMVArun2v1DBdR03oldDMwLT',
 'tau_byTightIsolationMVArun2v1DBnewDMwLT',
 'tau_byTightIsolationMVArun2v1DBoldDMwLT',
 'tau_byTightIsolationMVArun2v1PWdR03oldDMwLT',
 'tau_byTightIsolationMVArun2v1PWnewDMwLT',
 'tau_byTightIsolationMVArun2v1PWoldDMwLT',
 'tau_byVLooseIsolationMVArun2v1DBdR03oldDMwLT',
 'tau_byVLooseIsolationMVArun2v1DBnewDMwLT',
 'tau_byVLooseIsolationMVArun2v1DBoldDMwLT',
 'tau_byVLooseIsolationMVArun2v1PWdR03oldDMwLT',
 'tau_byVLooseIsolationMVArun2v1PWnewDMwLT',
 'tau_byVLooseIsolationMVArun2v1PWoldDMwLT',
 'tau_byVTightIsolationMVArun2v1DBdR03oldDMwLT',
 'tau_byVTightIsolationMVArun2v1DBnewDMwLT',
 'tau_byVTightIsolationMVArun2v1DBoldDMwLT',
 'tau_byVTightIsolationMVArun2v1PWdR03oldDMwLT',
 'tau_byVTightIsolationMVArun2v1PWnewDMwLT',
 'tau_byVTightIsolationMVArun2v1PWoldDMwLT',
 'tau_byVVLooseIsolationMVArun2v1DBoldDMwLT',
 'tau_byVVTightIsolationMVArun2v1DBdR03oldDMwLT',
 'tau_byVVTightIsolationMVArun2v1DBnewDMwLT',
 'tau_byVVTightIsolationMVArun2v1DBoldDMwLT',
 'tau_byVVTightIsolationMVArun2v1PWdR03oldDMwLT',
 'tau_byVVTightIsolationMVArun2v1PWnewDMwLT',
 'tau_byVVTightIsolationMVArun2v1PWoldDMwLT'
]


MU_FEATURES = [
    
]

EL_FEATURES = [
    
]




def select_gen_particle(event, index=0):
    """
    Function for the gen level particles selection:
    """
    pt = np.sqrt(root_file['genparticles_px'].array()[index]**2 + root_file['genparticles_py'].array()[index]**2 )
    pdg_id = root_file['genparticles_pdgid'].array()[index]
    promtTau = np.where(root_file['genparticles_isPromptTauDecayProduct'].array()[index]==1)[0]
    isPromt = np.where(root_file['genparticles_isPrompt'].array()[index]==1)[0]
    
    all_particles = np.where(isPromt>=0)[0]
    electrons = np.where(np.abs(pdg_id)==11)[0]
    
    muons = np.where(np.abs(pdg_id)==13)[0]
    
    #Electrons and Muons:
    
    index  = np.intersect1d(promtTau, np.where(pt>8)[0]) 
    index_1 = np.intersect1d(isPromt, np.where(pt>8)[0]) 
    
    if index.shape[0] == 0:
        el_index =  np.intersect1d(index_1, electrons)
        mu_index =  np.intersect1d(index_1, muons)
    elif index_1.shape[0] == 0 :
        el_index =  np.intersect1d(index, electrons)
        mu_index =  np.intersect1d(index, muons)
    else:
        el_index = np.array([])
        mu_index = np.array([])
    
        
    #Tau
    tau_index = np.setdiff1d(isPromt,el_index)
    tau_index = np.setdiff1d(tau_index, mu_index)
    tau_index  = np.where(np.sum(pt[tau_index])>15)[0]
    
    #Jets:
    jet_index =  np.setdiff1d(all_particles, el_index)
    jet_index =  np.setdiff1d(jet_index, mu_index)
    jet_index =  np.setdiff1d(jet_index, tau_index)

    return {'tau':tau_index, 'muon':mu_index, 'el':el_index, 'jet':jet_index}


def calc_phi(px, py):
    """
    Calculate Phi
    """
    eta = []
    for i in range(px.shape[0]):
        _px, _py = px[i], py[i]
        if _px==0 and _py >0:_eta = PI
        elif _px==0 and _py <0:_eta = -PI
        elif _px>0:_eta = np.arctan(py/px)
        elif _px<0 and _py >0:_eta = np.arctan(py/px) + PI
        elif _px<0 and _py <0:_eta = np.arctan(py/px)  - PI
        else:_eta = 0
        eta.append(_eta)
    return np.array(eta)

def calc_eta(px, py, pz):
    """
    Calculate Eta
    """
    ratio = pz/np.sqrt(px**2 + py**2)
    _phi = np.arctan(ratio)
    return _phi


def calculate_matching(root_file, indexes, event_index):
    """
	Calculate the matching between particle and gen level particle;    
    """
    PI=3.14
    muon_index = indexes['muon']
    electron_index = indexes['el']
    tau_index = indexes['tau']
    jet_index = indexes['jet']

    MatchedTau = []
    MatchedEl   =  []
    MatchedMu = []
    MatchedJet  = []

    gen_px = root_file['genparticles_px'].array()[event_index]
    gen_py = root_file['genparticles_py'].array()[event_index]
    gen_pz = root_file['genparticles_py'].array()[event_index]

    dR = 0
    for tau in range(root_file['tau_count'].array()[event_index]):
        
        if np.sqrt(root_file['tau_px'].array()[event_index][tau]**2 + root_file['tau_px'].array()[event_index][tau]**2)<20 or np.abs(root_file['tau_eta'].array()[event_index][tau])>2.3:
            continue
        eta, phi  = root_file['tau_eta'].array()[event_index][tau], root_file['tau_phi'].array()[event_index][tau]
     
        #Muon matching:
        if muon_index.shape[0]>0:
            _gen_phi = calc_phi(gen_px[muon_index], gen_py[muon_index])
            _gen_eta = calc_eta(gen_px[muon_index], gen_py[muon_index], gen_pz[muon_index] )
            dR = np.sqrt( (_gen_eta-eta)**2 + (_gen_phi-phi)**2)
            try:MatchedMu.append(dR)
            except Exception:1000#MatchedMu.append(dR)

        
        #Electron matching:
        if electron_index.shape[0]>0:
            _gen_phi = calc_phi(gen_px[electron_index], gen_py[electron_index])
            _gen_eta = calc_eta(gen_px[electron_index], gen_py[electron_index], gen_pz[electron_index] )
            dR = np.sqrt( (_gen_eta-eta)**2 + (_gen_phi-phi)**2)
            try: MatchedEl.append(dR)
            except Exception: 1000#MatchedEl.append(dR)


        #Tau matching: 
        if tau_index.shape[0]>0:
            _gen_phi = calc_phi(gen_px[tau_index], gen_py[tau_index])
            _gen_eta = calc_eta(gen_px[tau_index], gen_py[tau_index], gen_pz[tau_index] )
            dR = np.sqrt((_gen_eta-eta)**2 + (_gen_phi-phi)**2)
            try:MatchedTau.append(dR)
            except Exception:1000#MatchedTau.append(dR)

        #Jet matching:    
        _gen_phi = calc_phi(gen_px[jet_index], gen_py[jet_index])
        _gen_eta = calc_eta(gen_px[jet_index], gen_py[jet_index], gen_pz[jet_index] )
        dR = np.sqrt( (_gen_eta-eta)**2 + (_gen_phi-phi)**2)
        
        try:MatchedJet.append(dR)
        except Exception: 1000#MatchedJet.append(dR)
    
    dR_list = [MatchedEl, MatchedMu, MatchedTau, MatchedJet]
    print(dR_list)
    #for instance in range(len(dR_list)):
    #    try:np.min(dR_list[instance])
    #    except Exception:dR_list[instance]  = 10000
    #    else:dR_list[instance] = np.min(dR_list[instance])
    
    #min_index = np.argmin(dR_list)
    #label = np.argmin(dR_list)
    #if dR_list[label]<0.2:return  np.argmin(dR_list), np.argmin(dR_list)
    #else:
    return dR_list


LABELS = []
DR = []
TauCounts = []

for i in [np.random.randint(root_file.numentries) for i in range(10)]:#1000]:
    indexes = select_gen_particle(root_file, index=i)
    dR_list = calculate_matching(root_file, indexes, event_index=i)
    #LABELS.append(label)
    DR.append(dR_list)
    TauCounts.append(root_file['tau_count'].array()[i])
    LABELS.append(root_file['tau_pt'].array()[i])
    break
    #print("MatchedMu: ", MatchedMu)
    #print("MatchedEl: ", MatchedEl)
    #print("MatchedJet: ", MatchedJet)
    #df_label.LABELS.value_counts()



