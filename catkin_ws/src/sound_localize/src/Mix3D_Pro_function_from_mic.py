# -*- coding: utf-8 -*-
"""
Thanks God, everything is good in him

Mix3D_Pro_function_from_mic function
"""

import numpy as np
import math
import pdb

def Mix3D_Pro_function_from_mic(Mic):
    # parameters
    #c          = 343
    fs         = 16000
    mic_size   = np.shape(Mic)
    M          = mic_size[0]
    SorLen     = mic_size[1]

    # Windowing
    NWIN       = 2048
    hopsize    = NWIN/2
    NumOfFrame = int(2*math.floor(SorLen/NWIN)-1)


    weight     = np.hanning(NWIN+1)
    w          = weight[0:NWIN]
    w.shape    = (NWIN,1)
    w          = np.transpose(w)

    #FFT
    NFFT       = NWIN
    df         = fs/NFFT
    Freqs      = np.zeros([1,int(NFFT/2)],dtype = float)

    for i in range(int(NFFT/2)):
        Freqs[0,i] = i*df

    # parameters of FFT
    source_win = np.zeros([M,NWIN],dtype = float)
    source_zp  = np.zeros([M,NWIN],dtype = float)
    SOURCE     = np.zeros([M,NFFT],dtype = complex)
    SOURCE_half= np.zeros([M,int(NFFT/2)],dtype = complex)
    P_half     = np.zeros([M,int(NFFT/2),NumOfFrame],dtype = complex)

    for FrameNo in range(NumOfFrame):
        # --time segment--
        t_start=(FrameNo)*hopsize

        # --transform to frequency domain--
        for ss in range(M):
            source_win[ss,:] = np.multiply(Mic[ss,int(t_start+0):int(t_start+NWIN)], w)
            source_zp[ss,:]  = source_win[ss,:]
            SOURCE[ss,:]     = np.fft.fft(source_zp[ss,:])
            SOURCE_half[ss,:]= SOURCE[ss,0:int(NFFT/2)]

        for ff in range(int(NFFT/2)):
            P_half[:,ff,FrameNo]=SOURCE_half[:,ff]

    return P_half
