
#Author: Dr. Rami Khushaba
# This function computes the spatial-temporal features benefiting Dynamic Time warping method for extracting spatial components
# and an LSTM based paradigm for temporal information.


import numpy as np
import scipy.io
import math
import numpy
import itertools
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw

#signal_emg = scipy.io.loadmat('emg_ex.mat')
#signal = signal_emg['emg_train_0_6']
#print(signal.shape)

def STW(input_signal,winsize,wininc):

    nw = math.floor((len(input_signal) - winsize) / (wininc)) + 1
    winsize = winsize
    wininc = wininc
    st = 0
    en = winsize
    steps = 1
    datasize = len(input_signal)
    Nsignals = len(input_signal[0])

    nn = (Nsignals * Nsignals - Nsignals) / 2 * 3
    nnint = int(nn)
    feat = np.zeros((nw, nnint))
    NUM = np.zeros((1, len(feat[0])))
    Beta = 0.75

    def extractTDfeatures(input):
        N = len(input)
        K = len(input[0])

        Sdx = numpy.diff(input, axis=0)
        Sddx = numpy.diff(Sdx, axis=0)

        b = list(range(1, K + 1))
        c = list(itertools.combinations(b, 2))
        comb = numpy.array(c)

        Feat = []
        Featd = []
        Featdd = []
        for k in range(120):
            curwin_comb_1 = input[:, comb[k, 0] - 1]
            curwin_comb_2 = input[:, comb[k, 1] - 1]

            Sdx_comb_1 = Sdx[:, comb[k, 0] - 1]
            Sdx_comb_2 = Sdx[:, comb[k, 1] - 1]

            Sddx_comb_1 = Sddx[:, comb[k, 0] - 1]
            Sddx_comb_2 = Sddx[:, comb[k, 1] - 1]

            cc1 = curwin_comb_1[np.newaxis]
            cc2 = curwin_comb_2[np.newaxis]

            sdc1 = Sdx_comb_1[np.newaxis]
            sdc2 = Sdx_comb_2[np.newaxis]

            sddc1 = Sddx_comb_1[np.newaxis]
            sddc2 = Sddx_comb_2[np.newaxis]

            distance1, path1 = fastdtw(cc1, cc2, dist=euclidean)
            distance2, path2 = fastdtw(sdc1, sdc2, dist=euclidean)
            distance3, path3 = fastdtw(sddc1, sddc2, dist=euclidean)
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

#STW_FE = STW(signal,30,10)
#print(STW_FE)








