# -*- coding: utf-8 -*-
"""
Thanks God, everything is good in him

MUSIC_location_theta_phi(SorNum,MicPos,P_half)
"""
import numpy as np
from numpy import linalg as LA
import math
import pdb

def MUSIC_location_theta_phi(SorNum,MicPos,P_half):
    c          = 343
    f          = 16000
    NWIN       = 2048
    deg2rad    = 0.017453

    weight     = np.hanning(NWIN+1)
    w          = weight[0:NWIN]
    w.shape    = (NWIN,1)
    w          = np.transpose(w)

    mic_shape  = np.shape(MicPos)
    NumofSensor= mic_shape[1]

    p_half_shape = np.shape(P_half)
    NumOfFrame   = p_half_shape[2]

    ang_resol  = 30
    yang_resol = 30

    #FFT
    NFFT       = NWIN
    df         = f/NFFT
    freqRange  = np.zeros([1,int(NFFT/2)],dtype = float)
    Ang        = np.zeros([1,int(360/ang_resol)],dtype = float)
    Yang       = np.zeros([1,int(90/yang_resol)],dtype = float)
    h          = np.zeros([p_half_shape[1],p_half_shape[0],p_half_shape[2]],dtype = complex)
    W          = np.zeros([1,int(NFFT/2)],dtype = float)
    k          = np.zeros([1,int(NFFT/2)],dtype = float)
    x_1        = np.zeros([NumofSensor,NumOfFrame],dtype = complex)
    Rxx        = np.zeros([NumofSensor,NumofSensor],dtype = complex)
    P          = np.zeros([205,int(90/yang_resol),int(360/ang_resol)],dtype = float)
    UnitVector = np.zeros([1,3],dtype = float)
    SteeringVector = np.zeros([1,NumofSensor],dtype = complex)
    Location   = np.zeros([1,2],dtype = float)
    a_rr_us    = np.zeros([NumofSensor,1],dtype = complex)

    for i in range(int(NFFT/2)):
        freqRange[0,i] = i*df

    for i in range(int(360/ang_resol)):
        Ang[0,i] = i*ang_resol

    for i in range(int(90/yang_resol)):
        Yang[0,i] = i*yang_resol

    for FrameNo in range(NumOfFrame):
        for ff in range(int(NFFT/2)):
            h[ff,:,FrameNo]=P_half[:,ff,FrameNo]

    for i in range(int(NFFT/2)):
        W[0,i] = 2*math.pi*freqRange[0,i]

    for i in range(int(NFFT/2)):
        k[0,i] = W[0,i]/c


    # calculate  theta(i) steering Vector  and Rxx matrix to do EVD
    for ScanFrequency in range(19,223):
         x_1[:,:]=h[ScanFrequency,:,0:NumOfFrame]
         Rxx[:,:]=x_1[:,:].dot(np.conj(x_1[:,:]).T) / float(NumOfFrame) #scan frequency
         D, V = LA.eig(Rxx[:,:])

         index = sorted(range(len(D)), key=lambda k: D[k], reverse=True)
         V_sort = V[:,index]

         if SorNum == 1:
             US=V_sort[:,0]
         else:
             US=V_sort[:,0:SorNum]

         for ind_us in range(5):
             a_rr_us[ind_us,0] = US[ind_us]
         PN = np.eye(NumofSensor, dtype=float)-a_rr_us.dot(np.conj(a_rr_us).T)

         # calculate  point in the map by mic weight and steering Vector between point in the map  and array
         # steering matrixs
         for theta in range(int(360/ang_resol)):
             for phi in range(int(90/yang_resol)):
                 UnitVector[0,0] = math.cos(Ang[0,theta]*deg2rad)*math.cos(Yang[0,phi]*deg2rad)
                 UnitVector[0,1] = math.sin(Ang[0,theta]*deg2rad)*math.cos(Yang[0,phi]*deg2rad)
                 UnitVector[0,2] = math.sin(Yang[0,phi]*deg2rad)
                 vector_mul      = UnitVector.dot(MicPos)
                 for ind_steer in range(len(vector_mul.T)):
                     SteeringVector[0,ind_steer]  = math.cos(k[0,ScanFrequency]*vector_mul[0,ind_steer])+1j*math.sin(k[0,ScanFrequency]*vector_mul[0,ind_steer])

                 SteeringVector_trans = SteeringVector.T
                 P[ScanFrequency-19,phi,theta] =abs(1/(np.conj(SteeringVector_trans).T.dot(PN).dot(SteeringVector_trans)))


    P_AA = np.mean(P, axis=0)
    P_AA_norm = P_AA/np.max(np.max(P_AA))
    max_value = np.max(np.max(P_AA_norm))
    result    = np.where(P_AA_norm == max_value)

    Location[0,0] = result[1]*30
    Location[0,1] = result[0]*30

    return Location
