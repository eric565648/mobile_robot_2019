# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import math
from snr_seg_one import snr_seg_one
import pdb


def tra_process_new_th(y):

    y_len                 = len(y)

    hann_len              = 1024
    frame_number          = int(2*math.floor(y_len/hann_len)-1)


    signal_seg=np.zeros([frame_number,hann_len])
    y1_extend_sig         = y.transpose()

    w_nn                  = np.hanning(hann_len+1)
    w_nn                  = w_nn[0:hann_len]
    n_fft                 = 1*hann_len

    u0 = 1
    s = 1


    H_W = np.zeros([1,int(0.5*n_fft+1)])

    sigma_d = np.zeros([int(0.5*n_fft+1),frame_number])

    alpha_d = np.zeros([int(0.5*n_fft+1),frame_number])

    gamma_d= np.zeros([int(0.5*n_fft+1),frame_number])

    beta_d = 0.6

    gamma_dy = np.zeros([int(0.5*n_fft+1),frame_number])

    seg_snr_vector = np.zeros([1,frame_number])

    PSN_predict = np.zeros([int(0.5*n_fft+1),frame_number])

    S_filter_full = np.zeros([1,hann_len],dtype = complex)
    signal_syn = np.zeros([1,y_len])
    mean_sigma_d = np.zeros([1,int(0.5*n_fft+1)])
    yd_ratio     = np.zeros([1,int(0.5*n_fft+1)])
    alpha_denominator = np.zeros([1,int(0.5*n_fft+1)])

    for segment in range(frame_number):

          y1_w = 0.5*np.multiply(y1_extend_sig[int(0+(segment)*hann_len*0.5):int(hann_len+(segment)*hann_len*0.5)], w_nn.transpose())

          y1_w_e = y1_w

          y1_freq = np.fft.fft(y1_w_e)/n_fft

          y1_freq = y1_freq[0:int(0.5*n_fft+1)]

          PSN = np.square(abs(y1_freq))


          if segment <= 9:
              sigma_d[:,segment] = PSN
              gamma_d[:,segment] = 0
              gamma_dy[:,segment] = 0.001

          else:
              for index2 in range(int(0.5*n_fft+1)):
                  mean_sigma_d[0,index2] = 0.1 * (sigma_d[index2,segment-1] + sigma_d[index2,segment-2] + sigma_d[index2,segment-3] + sigma_d[index2,segment-4]+\
                               sigma_d[index2,segment-5] + sigma_d[index2,segment-6] + sigma_d[index2,segment-7] + sigma_d[index2,segment-8] + sigma_d[index2,segment-9]+\
                               sigma_d[index2,segment-10])

              for index in range(int(0.5*n_fft+1)):
                  if segment == 0:
                      PSN_predict[index,segment] = PSN[index]
                  else:
                      PSN_predict[index,segment] = 0.5 * PSN_predict[index,segment] + 0.5 * PSN[index]

              gamma_d[:,segment] = PSN / mean_sigma_d

              for index in range(int(0.5*n_fft+1)):
                   alpha_denominator[0,index] = 1 + math.pow(2.718,(-beta_d * (gamma_d[index,segment]-1.5)))


              alpha_d[:,segment] = 1 / alpha_denominator

              sigma_d[:,segment] = np.multiply(alpha_d[:,segment],sigma_d[:,segment-1])+np.multiply((1-alpha_d[:,segment]),PSN.transpose())

              for index in range(int(0.5*n_fft+1)):
                   if sigma_d[index,segment] > 0:
                      yd_ratio[0,index] = PSN_predict[index,segment] / sigma_d[index,segment].transpose()
                   else:
                      yd_ratio[0,index] = 1000

              gamma_dy[:,segment] = (0.5) * yd_ratio[0,:]

          for index in range(int(0.5*n_fft+1)):
                if segment <= 9:

                    H_W[0,index] = gamma_dy[index,segment] / (10 + gamma_dy[index,segment])

                else:
                    um = u0 - (seg_snr_vector[0,segment-1] / s)

                    if um >= 0.01:
                        u = um
                    else:
                        u = 0.01

                    H_W[0,index] = gamma_dy[index,segment] / (1*u + gamma_dy[index,segment])

          S_filter = np.multiply(y1_freq , H_W)


          a_middle = S_filter[0,int(0.5*n_fft)].real

          S_filter_full[0,int(0.5*n_fft)] = a_middle


          for index in range(int(0.5*n_fft)):
              S_filter_full[0,index] = (S_filter[0,index])



          for index in range(int(0.5*n_fft-1)):
              S_filter_full[0,hann_len-index-1] = np.conj(S_filter_full[0,index+1])

          s_test   =  S_filter_full
          s_filter_o = np.fft.ifft(s_test)


          signal_seg[segment,0:hann_len] = s_filter_o[0:hann_len].real

          segSNR = snr_seg_one(s_filter_o[0:hann_len].T, y1_w, hann_len)

          seg_snr_vector[0,segment] = segSNR


    for index_n in range(frame_number):
        signal_syn[0,int(0+(index_n)*hann_len*0.5):int(hann_len+(index_n)*hann_len*0.5)] = signal_syn[0,int(0+(index_n)*hann_len*0.5):int(hann_len+(index_n)*hann_len*0.5)] + signal_seg[index_n,0:hann_len]

    max_val      = np.amax(abs(signal_syn))
    min_val      = np.amin(abs(signal_syn))
    gth          = 0.1*(max_val-min_val)+min_val

    for index in range(y_len):
        if abs(signal_syn[0,index]) < gth:
           signal_syn[0,index] = 1*signal_syn[0,index]

    max_val = np.amax(abs(signal_syn))

    # signal_syn = signal_syn / max_val;

    xo = signal_syn

    return xo
