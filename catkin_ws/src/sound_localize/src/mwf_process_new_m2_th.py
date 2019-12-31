# -*- coding: utf-8 -*-
"""
Thanks God, God created the heavens and the earth
Created on Thu Nov 21 09:47:43 2019
@author: TEA
"""

import numpy as np
from   numpy.linalg import inv, pinv
import math
from tra_process_new_th import tra_process_new_th
import pdb

def mwf_process_new_m2_th(y1,y2,Fs1):

    #y_mic1_tra            = np.zeros([1,16000])
    #y_mic2_tra            = np.zeros([1,16000])

    #for ind in range(16000):
    #    y_mic1_tra[0,ind] = y1[ind]#tra_process_new_th(y1)
    #    y_mic2_tra[0,ind] = y2[ind]#tra_process_new_th(y2)

    y_mic1_tra            = tra_process_new_th(y1)
    y_mic2_tra            = tra_process_new_th(y2)

    #print("y mic1: ", y_mic1_tra)
    #print("y mic2: ", y_mic2_tra)

    y_len                 = len(y_mic1_tra.T)
    #print("y len: ", y_len)

    hann_len               = 1024
    n_fft                  = 1*hann_len

    frame_number           = int(2*math.floor(y_len/hann_len)-1)
    # print("frame num: ", frame_number)
    d_s                    = 0.05
    D_mat                  = np.zeros([2,2])
    D_mat[0,0]             = 0
    D_mat[0,1]             = d_s
    D_mat[1,0]             = d_s
    D_mat[1,1]             = 0
    c_speed                = 343

    a                      = np.zeros([2,1])
    a[0,0]                 = 1
    a[1,0]                 = 1


    weight                 = np.hanning(hann_len+1)
    w                      = weight[0:hann_len]
    w.shape                = (hann_len,1)
    w                      = np.transpose(w)

    x_syn                  = np.zeros([hann_len,frame_number])

    sigma_d1               = np.zeros([int(0.5*n_fft+1),frame_number])
    alpha_d1               = np.zeros([int(0.5*n_fft+1),frame_number])
    gamma_d1               = np.zeros([int(0.5*n_fft+1),frame_number])

    sigma_d2               = np.zeros([int(0.5*n_fft+1),frame_number])
    alpha_d2               = np.zeros([int(0.5*n_fft+1),frame_number])
    gamma_d2               = np.zeros([int(0.5*n_fft+1),frame_number])

    mean_sigma_d1          = np.zeros([1,int(0.5*n_fft+1)])
    mean_sigma_d2          = np.zeros([1,int(0.5*n_fft+1)])

    S_filter_full          = np.zeros([1,n_fft],dtype = complex)
    alpha_denominator1     = np.zeros([1,int(0.5*n_fft+1)])
    alpha_denominator2     = np.zeros([1,int(0.5*n_fft+1)])
    s_syn                  = np.zeros([1,y_len])

    beta_d = 0.6


    for frame_ind in range(frame_number):

        y_mic1_zero = np.multiply(y_mic1_tra[0,int(0+(frame_ind)*0.5*hann_len):int(hann_len+(frame_ind)*0.5*hann_len)], w)
        y_mic2_zero = np.multiply(y_mic2_tra[0,int(0+(frame_ind)*0.5*hann_len):int(hann_len+(frame_ind)*0.5*hann_len)], w)

        y_mic1_fft  = np.fft.fft(y_mic1_zero)
        y_mic2_fft  = np.fft.fft(y_mic2_zero)

        y_mic1_fft = y_mic1_fft[0,0:int(0.5*n_fft+1)]
        y_mic2_fft = y_mic2_fft[0,0:int(0.5*n_fft+1)]

        y_mat       = np.array([[y_mic1_fft],[y_mic2_fft]])
        #print("y mat: ", y_mat)
        y_power_mat = np.array([[np.square(abs(y_mic1_fft))],[np.square(abs(y_mic2_fft))]])

        PSN1        = np.square(abs(y_mic1_fft))
        PSN2        = np.square(abs(y_mic2_fft))

        if frame_ind >= 10:
            for index2 in range(int(0.5*n_fft+1)):
                mean_sigma_d1[0,index2] = 0.1 * (sigma_d1[index2,frame_ind-1] + sigma_d1[index2,frame_ind-2] + sigma_d1[index2,frame_ind-3] + sigma_d1[index2,frame_ind-4]+\
                              sigma_d1[index2,frame_ind-5] + sigma_d1[index2,frame_ind-6] + sigma_d1[index2,frame_ind-7] + sigma_d1[index2,frame_ind-8] + sigma_d1[index2,frame_ind-9]+\
                              sigma_d1[index2,frame_ind-10])

                mean_sigma_d2[0,index2] = 0.1 * (sigma_d2[index2,frame_ind-1] + sigma_d2[index2,frame_ind-2] + sigma_d2[index2,frame_ind-3] + sigma_d2[index2,frame_ind-4]+\
                              sigma_d2[index2,frame_ind-5] + sigma_d2[index2,frame_ind-6] + sigma_d2[index2,frame_ind-7] + sigma_d2[index2,frame_ind-8] + sigma_d2[index2,frame_ind-9]+\
                              sigma_d2[index2,frame_ind-10])


            gamma_d1[:,frame_ind] = PSN1 / mean_sigma_d1
            for index in range(int(0.5*n_fft+1)):
                   alpha_denominator1[0,index] = 1 + math.pow(2.718,(-beta_d * (gamma_d1[index,frame_ind]-1.5)))

            alpha_d1[:,frame_ind] = 1 / alpha_denominator1
            sigma_d1[:,frame_ind] = np.multiply(alpha_d1[:,frame_ind], sigma_d1[:,frame_ind-1]) + np.multiply(1-alpha_d1[:,frame_ind], PSN1.T)

            gamma_d2[:,frame_ind] = PSN2 / mean_sigma_d2
            for index in range(int(0.5*n_fft+1)):
                   alpha_denominator2[0,index] = 1 + math.pow(2.718,(-beta_d * (gamma_d2[index,frame_ind]-1.5)))

            alpha_d2[:,frame_ind] = 1 / alpha_denominator2
            sigma_d2[:,frame_ind] = np.multiply(alpha_d2[:,frame_ind], sigma_d2[:,frame_ind-1]) + np.multiply((1-alpha_d2[:,frame_ind]), PSN2.T)



        else:

            sigma_d1[:,frame_ind] = PSN1
            gamma_d1[:,frame_ind] = 0

            sigma_d2[:,frame_ind] = PSN2
            gamma_d2[:,frame_ind] = 0


        sigma_d_mat = np.array([[sigma_d1[:,frame_ind].T],[sigma_d2[:,frame_ind].T]])
        for index in range(int(0.5*n_fft+1)):

            if frame_ind == 0:
                phi_y      = np.sqrt(y_power_mat[:,:,index].dot(y_power_mat[:,:,index].T) / n_fft)
                phi_v      = np.sqrt(sigma_d_mat[:,:,index].dot(sigma_d_mat[:,:,index].T) / n_fft)

            else:
                phi_y      = 0.5 * phi_y + 0.5 * np.sqrt(y_power_mat[:,:,index].dot(y_power_mat[:,:,index].T) / n_fft)
                phi_v      = 0.5 * phi_v + 0.5 * np.sqrt(sigma_d_mat[:,:,index].dot(sigma_d_mat[:,:,index].T) / n_fft)



            if frame_ind >= 10:
                gamma_mat = np.sinc(2*(index+1)*Fs1/(n_fft*c_speed)*D_mat)

                c_test = a.T.dot(phi_v).dot(a) / np.square(a.T.dot(a))

                A         = np.array([[np.square(a.T.dot(a))*c_test, a.T.dot(gamma_mat).dot(a)*c_test, a.T.dot(phi_v).dot(a)],[a.T.dot(gamma_mat).dot(a)*c_test, sum(np.diag(gamma_mat.T.dot(gamma_mat)))*c_test, sum(np.diag(gamma_mat.T.dot(phi_v)))],[a.T.dot(phi_v).dot(a), sum(np.diag(gamma_mat.T.dot(phi_v))), sum(np.diag(phi_v.T.dot(phi_v)))]],dtype='float')

                b     = np.array([[a.T.dot(phi_y).dot(a)],[sum(np.diag(phi_y.T.dot(gamma_mat)))],[sum(np.diag(phi_y.T.dot(phi_v)))]])

                p         = pinv(A).dot(b)

                phi_x     = p[0,0]
                phi_d     = p[1,0]
                u         = p[2,0]


                phi_dv    = phi_d*gamma_mat + u*phi_v

                if (sum(abs(np.diag(phi_dv))) > 0 and abs(np.linalg.det(phi_dv))) > 0:
                    w_mvdr    = (inv(phi_dv).dot(a))/(a.T.dot(inv(phi_dv)).dot(a))
                    zeta      = phi_x/inv(a.T.dot(inv(phi_dv)).dot(a))
                else:
                    w_mvdr = np.array([[1],[1]])
                    zeta      = 1.0

                w_wf      = zeta/(1.0+zeta)
                w_mwf     = w_mvdr*w_wf
                x_es_fft  = (w_mwf.T).dot(y_mat[:,0,:])

            else:
                w_mvdr = 0.001*np.array([[1],[1]])
                zeta      = 1.0
                w_wf      = zeta / (1.0+zeta)
                w_mwf     = w_mvdr*w_wf
                x_es_fft  =( w_mwf.T).dot(y_mat[:,0,:])


        # print("w_wf: ", w_wf)

        a_middle = (x_es_fft[0,int(0.5*n_fft)]).real
        S_filter_full[0,int(0.5*n_fft)] = a_middle

        # in order to get real number after IFFT, we do inverse order.
        for index in range(int(0.5*n_fft)):
            S_filter_full[0,index] = (x_es_fft[0,index])


        for index in range(int(0.5*n_fft-1)):
            S_filter_full[0,hann_len-index-1] = np.conj(S_filter_full[0,index+1])


        x_es = (np.fft.ifft(S_filter_full)).real
        x_syn[:,frame_ind] = x_es[0,0:hann_len]
        #print("x_es: ", x_es)


    for index in range(frame_number):
        s_syn[0,int(0+0.5*hann_len*(index)):int(hann_len+0.5*hann_len*(index))] = s_syn[0,int(0+0.5*hann_len*(index)):int(hann_len+0.5*hann_len*(index))] + x_syn[0:hann_len,index].T;


    max_val      = np.amax(abs(s_syn))
    min_val      = np.amin(abs(s_syn))
    gth          = 0.1*(max_val-min_val)+min_val

    for index in range(y_len):
        if abs(s_syn[0,index]) < gth:
           s_syn[0,index] = 1*s_syn[0,index]


    val_max   = 1 #np.amax(abs(s_syn))
    xo     = s_syn / val_max
    return xo
