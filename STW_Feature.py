
# Author: Milad Jabbari
# This function computes the spatial-temporal features benefiting Dynamic Time warping method for extracting spatial components
# and an LSTM based paradigm for temporal information.

# Inputs:
# input_signal:  columns of signals
# winsize: window size 
# wininc: increasement size

# Output:
# STW: Spatio-Temporal Warping (STW) Feature

# P.S: The code is Python version of the primary function written by Dr. Rami Kushaba in Matlab for the following paper:
# Jabbari, Milad, Rami Khushaba, and Kianoush Nazarpour. "Spatio-temporal warping for myoelectric control: an offline, feasibility study."
# Journal of Neural Engineering 18.6 (2021): 066028.  

# Reference:
# % References
# [1] R. N. Khushaba, A. Al-Ani, A. Al-Timemy, A. Al-Jumaily, "A Fusion of Time-Domain Descriptors for Improved Myoelectric Hand Control", ISCIT2016 Conference, Greece, 2016.
# [2] A. Al-Timemy, R. N. Khushaba, G. Bugmann, and J. Escudero, "Improving the Performance Against Force Variation of EMG Controlled Multifunctional Upper-Limb Prostheses for Transradial Amputees",
#     IEEE Transactions on Neural Systems and Rehabilitation Engineering, DOI: 10.1109/TNSRE.2015.2445634, 2015.
# [3] R. N. Khushaba, Maen Takruri, Jaime Valls Miro, and Sarath Kodagoda, "Towards limb position invariant myoelectric pattern recognition using time-dependent spectral features",
#     Neural Networks, vol. 55, pp. 42-58, 2014.
# [4] Jabbari, Milad, Rami Khushaba, and Kianoush Nazarpour. "Spatio-temporal warping for myoelectric control: an offline, feasibility study."
#     Journal of Neural Engineering 18.6 (2021): 066028.  


import numpy as np
import scipy.io
import math
import numpy
import itertools
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw

def STW(input_signal,winsize,wininc):
    
    ## Parameters Defining:
    nw = math.floor((len(input_signal) - winsize) / (wininc)) + 1
    winsize = winsize
    wininc = wininc
    steps = 1
    datasize = len(input_signal)
    Nsignals = len(input_signal[0])
    nn = (Nsignals * Nsignals - Nsignals) / 2 * 3
    nnint = int(nn)
    
    ## Memory Allocation:
    feat = np.zeros((nw, nnint))
    
    ## Cell State Parameters:
    NUM = np.zeros((1, len(feat[0])))
    Beta = 0.75
    
    ##
    st = 0
    en = winsize

    def extractTDfeatures(input):
        N = len(input)
        K = len(input[0])
        
        ## First and Secodn Derivative
        Sdx = numpy.diff(input, axis=0)
        Sddx = numpy.diff(Sdx, axis=0)

        b = list(range(1, K + 1))
        c = list(itertools.combinations(b, 2))
        comb = numpy.array(c)

        Feat = []
        Featd = []
        Featdd = []
        for k in range(len(comb)):
            curwin_comb_1 = input[:, comb[k, 0] - 1]
            curwin_comb_2 = input[:, comb[k, 1] - 1]

            Sdx_comb_1 = Sdx[:, comb[k, 0] - 1]
            Sdx_comb_2 = Sdx[:, comb[k, 1] - 1]

            Sddx_comb_1 = Sddx[:, comb[k, 0] - 1]
            Sddx_comb_2 = Sddx[:, comb[k, 1] - 1]

            distance1, path1 = fastdtw(curwin_comb_1[np.newaxis], curwin_comb_2[np.newaxis], dist=euclidean)
            distance2, path2 = fastdtw(Sdx_comb_1[np.newaxis], Sdx_comb_2[np.newaxis], dist=euclidean)
            distance3, path3 = fastdtw(Sddx_comb_1[np.newaxis], Sddx_comb_2[np.newaxis], dist=euclidean)
            Feat.append(distance1)
            Featd.append(distance2)
            Featdd.append(distance3)

        final_feat = numpy.array(Feat)
        final_featd = numpy.array(Featd)
        final_featdd = numpy.array(Featdd)

        Final_Feat = np.hstack((final_feat, final_featd, final_featdd))
        F_Feat = numpy.log1p(Final_Feat)
        # TD_Feat = F_Feat[np.newaxis]
        return F_Feat

    STW_Fe = []
    for j in range(nw):
        curwin = input_signal[st:en, :]
        featureCurWin = extractTDfeatures(curwin)

        if j > steps - 1:
            prevwin = input_signal[(st - steps * wininc):(en - steps * wininc), :]
            featureprevWin = extractTDfeatures(prevwin)
            Beta = 1.25
            feat = np.log(np.multiply(featureprevWin, featureCurWin) + NUM * Beta) + np.log(NUM)
        else:
            if NUM.sum() != 0:
                feat = np.log(np.multiply(featurCurWin, featureCurWin) + NUM * Beta) + np.log(NUM)
            else:
                feat = np.log(featureCurWin * 2)

        if j == 0:
            NUM = featureCurWin
        else:
            NUM = abs(NUM + featureCurWin)

        st = st + wininc
        en = en + wininc
        STW_Fe.append(feat)

    STW = numpy.array(STW_Fe)
    return STW
