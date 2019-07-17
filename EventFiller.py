import numpy as np
import math


class EventFiller:

    def __init__(self, evt):
        self.evt = evt

    def Fill(self, maxCands):

        if (self.evt['nPFCands_1'] == 0):
            return []
        try:
            sortedInds = np.argsort(self.evt['pfCandPt_1'])
        except:
            return []
        else:
            sortedInds = np.argsort(self.evt['pfCandPt_1'])
        # print self.evt
        pt = self.evt['pfCandPt_1']
        pz = self.evt['pfCandPz_1']

        ptrel = self.evt['pfCandPtRel_1']
        pzrel = self.evt['pfCandPzRel_1']

        dr = self.evt['pfCandDr_1']
        deta = self.evt['pfCandDEta_1']

        dphi = self.evt['pfCandDPhi_1']
        eta = self.evt['pfCandEta_1']

        dz = self.evt['pfCandDz_1']
        dzerr = self.evt['pfCandDzErr_1']
        dzsig = self.evt['pfCandDzSig_1']

        d0 = self.evt['pfCandD0_1']
        d0err = self.evt['pfCandD0Err_1']
        d0sig = self.evt['pfCandD0Sig_1']

        pixhits = self.evt['pfCandPixHits_1']
        hits = self.evt['pfCandHits_1']
        nlosthits = self.evt['pfCandLostInnerHits_1']

        d0d0 = self.evt['pfCandD0D0_1']
        dzdz = self.evt['pfCandDzDz_1']
        d0dz = self.evt['pfCandD0Dz_1']
        d0dphi = self.evt['pfCandD0Dphi_1']

        ptrelsq = self.evt['pfCandPtRelPtRel_1']

        puppi = self.evt['pfCandPuppiWeight_1']

        fromPv = self.evt['pfCandFromPV_1']
        vtxQual = self.evt['pfCandVtxQuality_1']
        trkPurity = self.evt['pfCandHighPurityTrk_1']
        isBarrel = np.abs(self.evt['pfCandEta_1'] * 2.75) < 1.4  # self.evt['pfCandIsBarrel_1']
        purity = self.evt['pfCandHighPurityTrk_1']
        charge = self.evt['pfCandCharge_1']
        pdgid = np.abs(self.evt['pfCandPdgid_1'])
        tauind = self.evt['pfCandTauIndMatch_1']
        vdx = self.evt['pfCandDVx_1']
        vdy = self.evt['pfCandDVy_1']
        vdz = self.evt['pfCandDVz_1']
        vd = self.evt['pfCandD_1']

        vdx = vdx * (charge != 0) + 1 * (charge == 0)
        vdy = vdy * (charge != 0) + 1 * (charge == 0)
        vdz = vdz * (charge != 0) + 1 * (charge == 0)
        vd = vd * (charge != 0) + 1 * (charge == 0)

        evt = np.c_[
            pt, pz, ptrel, pzrel, dr,
            deta, dphi, eta, dz, dzsig,
            d0, d0sig, dzerr, d0err, d0d0,
            charge == 0, charge == 1, charge == -1, pdgid > 22, pdgid == 22,
            # vdx	   , vdy	, vdz	      , vd	  ,
            dzdz, d0dz, d0dphi, ptrelsq, pixhits,
            hits, nlosthits == -1, nlosthits == 0, nlosthits == 1, nlosthits == 2,
            puppi, vtxQual == 1, vtxQual == 5, vtxQual == 6, vtxQual == 7,
            fromPv == 1, fromPv == 2, fromPv == 3, isBarrel, purity,
            pdgid == 1, pdgid == 2, pdgid == 11, pdgid == 13, pdgid == 130,
            pdgid == 211, tauind]



        evt = evt[sortedInds]
        evt = evt[len(evt) - maxCands:]
        evt[evt == np.nan] = 0
        evt[evt == np.inf] = 0
        if np.min(evt) == np.nan:
            return [0] * maxCands
        if np.abs(np.max(evt)) == np.inf:
            return [0] * maxCands

        return evt
