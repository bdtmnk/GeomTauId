""""
Writen by Leonid Didukh
Based on: https://energyflow.network/examples/
"""
import energyflow as ef
from energyflow.archs import PFN
from keras.utils import plot_model

def DeepSet(nParticles,nFeatures, Phi_sizes= (50, 50, 12), F_sizes=(50, 50, 50)):

    model = PFN(input_dim=nFeatures, Phi_sizes=Phi_sizes, F_sizes=F_sizes, output_dim=1, output_act='sigmoid', loss='binary_crossentropy')
    plot_model(model, to_file='deepset.png')
    return model


