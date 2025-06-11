import pandas as pd
import glob
import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import re
from tqdm import tqdm

filepath_results= "./results/"

eids2enzymes = pd.read_csv('../data/screening_data/ELYS_ID_to_Enzyme_name_VB.txt', sep='\t', encoding='utf8')
eids2enzymes = dict(zip(eids2enzymes.ELYS_ID, eids2enzymes.Enzyme_name))

d3to1 = {'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K',
'ILE': 'I', 'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N', 
'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W', 
'ALA': 'A', 'VAL':'V', 'GLU': 'E', 'TYR': 'Y', 'MET': 'M'}

ppm = 50.
score_threshold = 0.85
ppm_integrate = 10.
rt_semiwindow = 6. # seconds
rt_semiwindow_cutoff = 60. # seconds
rt_semiwindow_integrate = 6. # seconds

from matchms import Spectrum
from matchms.similarity import CosineGreedy
from matchms.filtering import *

def cosine_greedy_score(mz_peaks, i_peaks, mz_peaks_ref, i_peaks_ref):
    query = Spectrum(mz=np.array(mz_peaks, dtype=float), intensities=np.array(i_peaks, dtype=float), metadata={"precursor_mz":1.})
    query = normalize_intensities(query)
    query = reduce_to_number_of_peaks(query, n_required=1, n_max=50)
    
    reference = Spectrum(mz=np.array(mz_peaks_ref), intensities=np.array(i_peaks_ref), metadata={"precursor_mz":1.})
    reference = normalize_intensities(reference)
    reference = reduce_to_number_of_peaks(reference, n_required=5, n_max=50)
    
    cosine_greedy = CosineGreedy(tolerance=0.15, mz_power=0., intensity_power=1.0) # tolerance in Dalton
    if reference is None:
        return 0.
    else:
        score = cosine_greedy.pair(reference, query)
        return float(score['score'])
    

enzymes_inclusion = pd.read_csv('../data/screening_data/Enzymes_SS030623.csv', dtype={'Enzyme Name': str})
enzymes_inclusion = set(enzymes_inclusion['Enzyme Name'])