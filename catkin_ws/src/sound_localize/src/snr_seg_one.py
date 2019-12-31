# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 13:03:40 2019

@author: TEA
"""
import math
import pdb

def snr_seg_one(clean_signal, enhanced_signal, w_len):

    len_clean_signal = len(clean_signal);
    len_enhanced_signal = len(enhanced_signal);

    if len_clean_signal == len_enhanced_signal:
        frame_len = w_len
        M = len_clean_signal / frame_len
        coef = 10 / M
        sum_snr = 0
        for index in range(len_clean_signal):#index = 1:len_clean_signal
            if clean_signal[index] == 0:
                clean_signal[index] = 2.2E-16

        numerator = 0
        denominator = 0
        for index2 in range(frame_len):
            numerator = numerator + math.pow(clean_signal[index2].real,2)
            denominator = denominator + math.pow(clean_signal[index2].real - enhanced_signal[index2],2)
        sum_snr = sum_snr + math.log10(numerator/denominator)

        snr = coef * sum_snr
    else:
        snr = 0
#end
    return snr
