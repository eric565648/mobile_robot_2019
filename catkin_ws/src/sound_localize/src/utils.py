# -*- coding: utf-8 -*-
"""
Created on Sat Dec 21 15:12:45 2019

@author: TommyCheng
"""

import numpy as np
import math

from numpy.linalg import inv
from numpy.linalg import norm
import math
from numpy.linalg import solve, norm
from scipy.io.wavfile import read,write
#from snr_seg_one import snr_seg_one
from matplotlib import pyplot
from scipy.signal import find_peaks
import time

def UCAMic(radius,MicNumber):
    mic_theta           = np.arange(0, 2*np.pi, 2*np.pi/MicNumber)
    MicPos = radius*np.array([np.cos(mic_theta),np.sin(mic_theta),np.zeros(6)])
    return MicPos

def Mix_3D_pro_function_one_source_simulation(MicPos):
    ## transpose error ? +- for image part in weight
    Ts     = 4096/16000
    c      = 343.0
    fs     = 16000
    SorNum = 1
    SorLen = int(fs*Ts)
    sr,x1  = read('C:/Users/TommyCheng/Desktop/DARPA/audio/speech.wav')
    x1     = x1/2**15
    x1     = x1[5000:5000+SorLen]
    m,n    = MicPos.shape
    source =np.reshape(x1,[1,SorLen])
    SorPos = np.zeros((SorNum,2))
    kappa  = np.zeros((SorNum,3))
    for i in range(SorNum):
        theta       = 270
        phi         = 20
        SorPos[i,:] = np.array([theta,phi])
        kappa[i,:]  = np.array([np.cos(np.pi*theta/180.0)*np.cos(np.pi*phi/180.0),np.sin(np.pi*theta/180.0)*np.cos(np.pi*phi/180.0),np.sin(np.pi*phi/180.0)])

    NWIN       = int(2048)
    hopsize    = int(NWIN / 2)                               #50% overlap
    win        = np.hanning(NWIN+1)
    win        = win[:NWIN]
    ## FFT
    NFFT       = int(2048)
    df         = fs/NFFT
    NumOfFrame = int(2*np.floor(SorLen/NWIN)-1)
    Freqs      = np.arange(0,(NFFT/2)*df,df)
    P_half      = np.zeros((n,len(Freqs),NumOfFrame),dtype=complex)
    source_win  = np.zeros((SorNum,NWIN))
    source_zp   = np.zeros((SorNum,NFFT))
    SOURCE_half = np.zeros((SorNum,int(NFFT/2)),dtype=complex)
    A           = np.zeros((n,SorNum),dtype=complex)
    k = 2*np.pi*Freqs/c

    for frameNo in range(NumOfFrame):
        t_start = frameNo*hopsize
        tt      = np.arange(int(t_start),int(t_start+NWIN),1)
        for ss in range(SorNum):
            source_win[ss,:]  = np.multiply(source[ss,tt],win)
            source_zp[ss,:]   = np.concatenate((source_win[ss,:],np.zeros((NFFT-NWIN))))
            SOURCE_half[ss,:] = np.fft.fft(source_zp[ss,:],NFFT)[:int(NFFT/2)]

        for ff in range(len(Freqs)):
            for ss in range(SorNum):
                for mm in range(n):
                    A[mm,ss]     = np.exp(1j*k[ff]*np.dot(kappa[ss,:],MicPos[:,mm]))
            P_half[:,ff,frameNo] = np.divide(np.dot(A,SOURCE_half[:,ff]),SorNum)
    return P_half

def Mix_from_mic(MicPos,MicSignal):
    c      = 343.0
    fs     = 16000
    NWIN       = int(2048)
    hopsize    = int(NWIN / 2)                               #50% overlap
    win        = np.hanning(NWIN+1)
    win        = win[:NWIN]
    ## FFT
    NFFT       = int(2048)
    df         = fs/NFFT
    Freqs      = np.arange(0,(NFFT/2)*df,df)

    MicNumber,SorLen = MicSignal.shape
    NumOfFrame = int(2*np.floor(SorLen/NWIN)-1)
    P_half      = np.zeros((MicNumber,len(Freqs),NumOfFrame),dtype=complex)
    source_win  = np.zeros((MicNumber,NWIN))
    source_zp   = np.zeros((MicNumber,NFFT))
    SOURCE_half = np.zeros((6,int(NFFT/2)),dtype=complex)
    for frameNo in range(NumOfFrame):
        t_start = frameNo*hopsize
        tt      = np.arange(int(t_start),int(t_start+NWIN),1)
        for ss in range(MicNumber):
            source_win[ss,:]  = np.multiply(MicSignal[ss,tt],win)
            source_zp[ss,:]   = np.concatenate((source_win[ss,:],np.zeros((NFFT-NWIN))))
            SOURCE_half[ss,:] = np.fft.fft(source_zp[ss,:],NFFT)[:int(NFFT/2)]

        for ff in range(len(Freqs)):
             P_half[:,ff,frameNo]=SOURCE_half[:,ff]

    return P_half

# MUSIC Parameter
def MUSIC_Parameter(P_half,fs,SorNum,select_range):
    # Parameter
    MicNo=np.size(P_half,0)
    NumOfFreqs=np.size(P_half,1)
    NumOfFrame=np.size(P_half,2)
    NWIN=2048
    NFFT=NWIN
    df=fs/NFFT
    Freqs=np.linspace(0,(NFFT/2-1)*df,int(NFFT/2))
    c=343
    length_select_range=len(select_range)
    w=2*math.pi*Freqs[select_range]
    k=w/c

    # Transform
    x=np.zeros([NumOfFreqs,MicNo,NumOfFrame],dtype=complex)
    for FrameNo in range(0,NumOfFrame):
        for ff in range(0,NumOfFreqs):
            x[ff,:,FrameNo]=P_half[:,ff,FrameNo]

    # Rxx
    PN=np.zeros([MicNo,MicNo,length_select_range],dtype=complex)

    for ff in range(length_select_range):
        x_1=x[select_range[ff],:,:]
        Rxx=np.dot(x_1,x_1.conj().T)/NumOfFrame
        [eigenvalue,eigenvector]=np.linalg.eig(Rxx)
        sort_eigenvalue=np.argsort(abs(eigenvalue))
        sort_eigenvector=eigenvector[:,sort_eigenvalue]
        Us=sort_eigenvector[:,-SorNum:]
        PN[:,:,ff]=np.eye(MicNo)-np.dot(Us,Us.conj().T)

    return k,PN

# MUSIC Grid Search
def MUSIC_Grid_Search(MicPos,PN,k):
    # Grid search
    EstiAng=np.arange(0,360,5)
    EstiYAng=np.arange(0,90,5)
    y=np.zeros([len(EstiYAng),len(EstiAng)])

    for theta in range(0,len(EstiAng)):
        for phi in range(0,len(EstiYAng)):
            position=np.array([math.cos(EstiYAng[phi]/(180/math.pi))*math.cos(EstiAng[theta]/(180/math.pi)),math.cos(EstiYAng[phi]/(180/math.pi))*math.sin(EstiAng[theta]/(180/math.pi)),math.sin(EstiYAng[phi]/(180/math.pi))])
            y[phi,theta]=cost_MUSIC(position,MicPos,PN,k)
            
    MaxValue = np.max(np.max(y,axis=1),axis=0)
    location = np.argwhere(y==MaxValue)
    azimuth_angle = EstiAng[location[0][1]]
    elevation_angle =EstiYAng[location[0][0]]
    #plt.contourf(EstiAng, EstiYAng, y, 8, alpha=.75, cmap=plt.cm.hot)
    return azimuth_angle,elevation_angle

# Cost MUSIC
def cost_MUSIC(position,MicPos,PN,k):
    cost=0
    NumOfFreqs=len(k)
    for ff in range(NumOfFreqs):
    #for ff in range(6,200):
        w=np.exp(1j*k[ff]*np.dot(position,MicPos))
        cost+=abs(1/np.dot(np.dot(w.conj(),PN[:,:,ff]),w.T))
    return cost


def MUSIC_PSO_localization(P_half,MicPos,SorNum,select_range):
    c      = 343.0
    fs     = 16000
    NWIN       = int(2048)
    win        = np.hanning(NWIN+1)
    win        = win[:NWIN]
    ## FFT
    NFFT       = int(2048)
    df         = fs/NFFT
    Freqs      = np.arange(0,(NFFT/2)*df,df)
    length_select_range = len(select_range)

    inertia=0.6
    correction_factor=1.2
    correction_factor_group=1.6
    Rshare=50
    iterations=5
    particles=72
    swarm =np.zeros((particles,7))
    Ang = np.arange(0,360,30)
    Yang = np.arange(0,90,15)
    Distance_matrix = np.zeros((particles,particles))
    M=np.zeros((particles,particles))
    temp1=np.zeros(2)
    k = np.divide(2*np.pi*Freqs[select_range],c)
    PN=np.zeros((6,6,length_select_range),dtype=complex)

    counter=0
    for x in range(12):
        for y in range(6):
            swarm[counter,0:2]=np.array([Ang[x],Yang[y]])
            counter = counter + 1

    for ScanFrequency in range(length_select_range):
        #print(ScanFrequency)
        x_1 = P_half[:,select_range[ScanFrequency],:]
        Rxx =np.dot(x_1,np.conj(x_1.T))
        a, b = np.linalg.eig(Rxx)
        US = b[:,a.argsort()[::-1]][:,:SorNum]
        PN[:,:,ScanFrequency] = np.identity(6)-np.dot(US,np.conj(US.T))

    for iter  in range(iterations):
        for i in range(particles):
            for j in range(2):
                swarm[i,j] =swarm[i,j] + swarm[i,j+4]

                if j==0 and swarm[i,j] <= 0:
                    swarm[i,j]= np.random.randint(360)
                if j==0 and swarm[i,j]>= 360:
                    swarm[i,j]=np.random.randint(360)
                if j==1 and swarm[i,j]<=20:
                    swarm[i,j]=10+np.random.randint(60)
                if j==1 and swarm[i,j]>=80:
                    swarm[i,j] =10+ np.random.randint(60)
            kappa = np.array([np.cos(np.pi*swarm[i,0]/180.0)*np.cos(np.pi*swarm[i,1]/180.0),np.sin(np.pi*swarm[i,0]/180.0)*np.cos(np.pi*swarm[i,1]/180.0),np.sin(np.pi*swarm[i,1]/180.0)])
            P_out =cost_MUSIC(kappa,MicPos,PN,k)

            if P_out > swarm[i,6]:
                for q in range(2):
                    swarm[i,q+2] = swarm[i,q]
            swarm[i,6]= P_out
            for j in range(i,particles):
                if j==i:
                    Distance_matrix[i,j]= 0
                else:
                    Distance_matrix[i,j] = norm(swarm[i,0:2]-swarm[j,0:2])

                Distance_matrix[j,i]=Distance_matrix[i,j]

            for z  in range(particles):
                if Distance_matrix[i,z]<Rshare:
                    M[z,i]=1
                else:
                    M[z,i]=0

        Follow_matrix = np.zeros((particles,particles))

        for i in range(particles):
            Decision_matrix = np.multiply(swarm[:,6],M[:,i])
            fbest = np.argmax(Decision_matrix)
            Follow_matrix[i,fbest] = 1
            for q in range(2):
                swarm[i,q+4] = inertia*swarm[i,q+4]+correction_factor*np.random.rand()*(swarm[i,q+2]-swarm[i,q])+correction_factor_group*np.random.rand()*(swarm[fbest,q+2]-swarm[i,q])

        #best_swarm_index = np.argmax(np.sum(Follow_matrix,axis=0))
        peaks , _ =find_peaks(np.sum(Follow_matrix,axis=0))
        index = peaks[np.argsort(swarm[peaks,6])][-SorNum:]
        best_swarm = swarm[index[0],0:2]
        '''
        inertia = inertia*0.8
        correction_factor=correction_factor*0.8
        correction_factor_group=correction_factor_group*0.8
        '''
        '''
        if iter<=1:
            temp1 =best_swarm
        else:
            if norm(best_swarm-temp1) <1:
                break
            else:
                temp1=best_swarm
        '''

    return best_swarm[0],best_swarm[1]


def MUSIC_Localization_freqrange_grid_search(P_half,MicPos,SorNum,select_range):
    c      = 343.0
    fs     = 16000
    NWIN       = int(2048)
    ## FFT
    NFFT       = int(2048)
    df         = fs/NFFT
    Freqs      = np.arange(0,(NFFT/2)*df,df)
    length_select_range = len(select_range)
    # 360* 90 ->30sec  180*18->3sec
    Ang = np.arange(0,360,5)
    Yang = np.arange(0,90,10)
    P_out=np.zeros((9,72))

    k = np.divide(2*np.pi*Freqs[select_range],c)
    PN=np.zeros((6,6,length_select_range),dtype=complex)

    for ScanFrequency in range(length_select_range):
        #print(ScanFrequency)
        x_1 = P_half[:,select_range[ScanFrequency],:]
        Rxx =np.dot(x_1,np.conj(x_1.T))
        a, b = np.linalg.eig(Rxx)
        US = b[:,a.argsort()[::-1]][:,:SorNum]
        PN[:,:,ScanFrequency] = np.identity(6)-np.dot(US,np.conj(US.T))

    for theta  in range(len(Ang)):
        for phi in range(len(Yang)):
            kappa = np.array([np.cos(math.pi*Ang[theta]/180.0)*np.cos(math.pi*Yang[phi]/180.0),np.sin(math.pi*Ang[theta]/180.0)*np.cos(math.pi*Yang[phi]/180.0),np.sin(math.pi*Yang[phi]/180.0)])
            P_out[phi,theta] =cost_MUSIC(kappa,MicPos,PN,k)
    '''
    pyplot.figure()
    pyplot.contourf(Ang, Yang, P_out)
    pyplot.colorbar()
    pyplot.show()
    '''
    MaxValue = np.max(np.max(P_out,axis=1),axis=0)
    location = np.argwhere(P_out==MaxValue)
    azimuth_angle = Ang[location[0][1]]
    elevation_angle =Yang[location[0][0]]

    '''
    advance_Ang=np.arange(azimuth_angle-3,azimuth_angle+4,1);
    advance_phi=np.arange(elevation_angle-3,elevation_angle+4,1);


    for theta  in range(len(advance_Ang)):
        for phi in range(len(advance_phi)):
                kappa = np.array([np.cos(np.pi*advance_Ang[theta]/180.0)*np.cos(np.pi*advance_phi[phi]/180.0),np.sin(np.pi*advance_Ang[theta]/180.0)*np.cos(np.pi*advance_phi[phi]/180.0),np.sin(np.pi*advance_phi[phi]/180.0)])
                P_out_advance[phi,theta] =cost_MUSIC(kappa,MicPos,PN,k)

    MaxValue = np.max(np.max(P_out_advance,axis=1),axis=0)
    location = np.argwhere(P_out_advance==MaxValue)
    azimuth_angle = advance_Ang[location[0][1]]
    elevation_angle =advance_phi[location[0][0]]
    '''
    return azimuth_angle,elevation_angle


def Multi_MUSIC_Localization_freqrange_grid_search(P_half,MicPos,SorNum,select_range):
    c      = 343.0
    fs     = 16000
    NWIN       = int(2048)
    ## FFT
    NFFT       = int(2048)
    df         = fs/NFFT
    Freqs      = np.arange(0,(NFFT/2)*df,df)
    length_select_range = len(select_range)
    Ang = np.arange(0,360,5)
    Yang = np.arange(0,90,10)
    P_out=np.zeros((9,72))
    k = np.divide(2*np.pi*Freqs[select_range],c)
    PN=np.zeros((6,6,length_select_range),dtype=complex)
    for ScanFrequency in range(length_select_range):
        #print(ScanFrequency)
        x_1 = P_half[:,select_range[ScanFrequency],:]
        Rxx =np.dot(x_1,np.conj(x_1.T))
        a, b = np.linalg.eig(Rxx)
        US = b[:,a.argsort()[::-1]][:,:SorNum]
        PN[:,:,ScanFrequency] = np.identity(6)-np.dot(US,np.conj(US.T))

    for theta  in range(len(Ang)):
        for phi in range(len(Yang)):
            kappa = np.array([np.cos(np.pi*Ang[theta]/180.0)*np.cos(np.pi*Yang[phi]/180.0),np.sin(np.pi*Ang[theta]/180.0)*np.cos(np.pi*Yang[phi]/180.0),np.sin(np.pi*Yang[phi]/180.0)])
            P_out[phi,theta] =cost_MUSIC(kappa,MicPos,PN,k)

    MaxValue = np.max(np.max(P_out,axis=1),axis=0)
    location = np.argwhere(P_out==MaxValue)
    azimuth_angle = Ang[location[0][1]]
    elevation_angle =Yang[location[0][0]]
    if  location[0][1]<=4 or location[0][1]>=68:
        thetaselect = np.array([0,1,2,3,4,68,69,70,71])
    else:
        thetaselect = np.arange(location[0][1]-4,location[0][1]+5,1)
    if  location[0][0]>= 7 :
        phiselect = np.array([5,6,7,8])
    else:
        phiselect = np.arange(location[0][0]-2,location[0][0]+2,1)

    # zero masking arround best position to find second peak
    for theta  in thetaselect:
            for phi in phiselect:
                P_out[phi,theta]=0

    MaxValue2 = np.max(np.max(P_out,axis=1),axis=0)
    location2 = np.argwhere(P_out==MaxValue2)
    azimuth_angle2 = Ang[location2[0][1]]
    elevation_angle2 =Yang[location2[0][0]]

    return azimuth_angle,elevation_angle,azimuth_angle2,elevation_angle2



def MUSIC_Localization_freqrange_theta_given_phi(P_half,MicPos,SorNum,select_range,input_phi):
    c      = 343.0
    fs     = 16000
    NWIN       = int(2048)
    ## FFT
    NFFT       = int(2048)
    df         = fs/NFFT
    Freqs      = np.arange(0,(NFFT/2)*df,df)
    length_select_range = len(select_range)
    #phi_range = 5
    #phi_distance  = 1
    Ang = np.arange(0,360,3)
    #Yang = np.arange( input_phi - phi_range, input_phi + phi_range, phi_distance)
    P_out=np.zeros(120)
    k = np.divide(2*np.pi*Freqs[select_range],c)
    PN=np.zeros((6,6,length_select_range),dtype=complex)

    for ScanFrequency in range(length_select_range):
        #print(ScanFrequency)
        x_1 = P_half[:,select_range[ScanFrequency],:]
        Rxx =np.dot(x_1,np.conj(x_1.T))
        a, b = np.linalg.eig(Rxx)
        US = b[:,a.argsort()[::-1]][:,:SorNum]
        PN[:,:,ScanFrequency] = np.identity(6)-np.dot(US,np.conj(US.T))

    for theta  in range(len(Ang)):
        #for phi in range(len(Yang)):
            kappa = np.array([np.cos(np.pi*Ang[theta]/180.0)*np.cos(np.pi*input_phi/180.0),np.sin(np.pi*Ang[theta]/180.0)*np.cos(np.pi*input_phi/180.0),np.sin(np.pi*input_phi/180.0)])
            P_out[theta] =cost_MUSIC(kappa,MicPos,PN,k)

    MaxValue = np.max(P_out)
    location = np.argwhere(P_out==MaxValue)
    azimuth_angle = Ang[location[0]]
    #elevation_angle =Yang[location[0]

    return azimuth_angle


def MUSIC_Localization_freqrange_theta_constant_phi(P_half,MicPos,SorNum,select_range,phi):
    c      = 343.0
    fs     = 16000
    NWIN       = int(2048)
    ## FFT
    NFFT       = int(2048)
    df         = fs/NFFT
    Freqs      = np.arange(0,(NFFT/2)*df,df)
    length_select_range = len(select_range)
    Ang = np.arange(0,360,5)
    P_out=np.zeros(72)
    P_out_advance=np.zeros(7)
    k = np.divide(2*np.pi*Freqs[select_range],c)
    PN=np.zeros((6,6,length_select_range),dtype=complex)
    for ScanFrequency in range(length_select_range):
        #print(ScanFrequency)
        x_1 = P_half[:,select_range[ScanFrequency],:]
        Rxx =np.dot(x_1,np.conj(x_1.T))
        a, b = np.linalg.eig(Rxx)
        US = b[:,a.argsort()[::-1]][:,:SorNum]
        PN[:,:,ScanFrequency] = np.identity(6)-np.dot(US,np.conj(US.T))

    for theta  in range(len(Ang)):
        kappa = np.array([np.cos(np.pi*Ang[theta]/180.0)*np.cos(np.pi*phi/180.0),np.sin(np.pi*Ang[theta]/180.0)*np.cos(np.pi*phi/180.0),np.sin(np.pi*phi/180.0)])
        P_out[theta] =cost_MUSIC(kappa,MicPos,PN,k)

    '''
    pyplot.figure()
    pyplot.plot(P_out)
    pyplot.show()
    '''
    index = np.argmax(P_out)
    azimuth_angle = Ang[index]

    advance_Ang=np.arange(azimuth_angle-3,azimuth_angle+4,1);


    for theta2  in range(len(advance_Ang)):
        kappa = np.array([np.cos(np.pi*advance_Ang[theta2]/180.0)*np.cos(np.pi*phi/180.0),np.sin(np.pi*advance_Ang[theta2]/180.0)*np.cos(np.pi*phi/180.0),np.sin(np.pi*phi/180.0)])
        P_out_advance[theta2] =cost_MUSIC(kappa,MicPos,PN,k)

    index = np.argmax(P_out_advance)
    azimuth_angle_advance = advance_Ang[index]

    return azimuth_angle,azimuth_angle_advance

def EVD_criterion(eigenvalue,baseline):
    eigenvalue=eigenvalue[eigenvalue.argsort()[::-1]]
    if np.real(eigenvalue[0]) >= baseline:
        flag=True
    else:
        flag=False

    if flag ==True:
        counter = 1
        for i in range(1,6,1):
            if  np.real(eigenvalue[0]-eigenvalue[i]) <= (1/5)*np.real(eigenvalue[0]):
                counter = counter + 1
            else:
                counter = counter
    else:
        counter = 0

    return counter

def Multi_MUSIC_Localization_freqrange_theta_constant_phi(P_half,MicPos,SorNum,select_range,phi):
    c      = 343.0
    fs     = 16000
    NWIN       = int(2048)
    ## FFT
    NFFT       = int(2048)
    df         = fs/NFFT
    Freqs      = np.arange(0,(NFFT/2)*df,df)
    length_select_range = len(select_range)
    Ang = np.arange(0,360,5)
    P_out=np.zeros(72)
    P_out_advance=np.zeros(7)
    P_out_advance2=np.zeros(7)
    k = np.divide(2*np.pi*Freqs[select_range],c)
    PN=np.zeros((6,6,length_select_range),dtype=complex)
    for ScanFrequency in range(length_select_range):
        #print(ScanFrequency)
        x_1 = P_half[:,select_range[ScanFrequency],:]
        Rxx =np.dot(x_1,np.conj(x_1.T))
        a, b = np.linalg.eig(Rxx)
        US = b[:,a.argsort()[::-1]][:,:SorNum]
        PN[:,:,ScanFrequency] = np.identity(6)-np.dot(US,np.conj(US.T))

    for theta  in range(len(Ang)):
        kappa = np.array([np.cos(np.pi*Ang[theta]/180.0)*np.cos(np.pi*phi/180.0),np.sin(np.pi*Ang[theta]/180.0)*np.cos(np.pi*phi/180.0),np.sin(np.pi*phi/180.0)])
        P_out[theta] =cost_MUSIC(kappa,MicPos,PN,k)
    '''
    pyplot.figure()
    pyplot.plot(P_out)
    pyplot.show()
    '''
    peaks , _ =find_peaks(P_out)
    index = peaks[np.argsort(P_out[peaks])][-SorNum:]

    if  Ang[index[0]] <=  Ang[index[1]]:
        azimuth_angle= Ang[index[0]]
        azimuth_angle2 = Ang[index[1]]
    else:
        azimuth_angle= Ang[index[1]]
        azimuth_angle2 = Ang[index[0]]

    advance_Ang=np.arange(azimuth_angle-3,azimuth_angle+4,1);
    advance_Ang2=np.arange(azimuth_angle2-3,azimuth_angle2+4,1);


    for theta2  in range(len(advance_Ang)):
        kappa = np.array([np.cos(np.pi*advance_Ang[theta2]/180.0)*np.cos(np.pi*phi/180.0),np.sin(np.pi*advance_Ang[theta2]/180.0)*np.cos(np.pi*phi/180.0),np.sin(np.pi*phi/180.0)])
        P_out_advance[theta2] =cost_MUSIC(kappa,MicPos,PN,k)
    index = np.argmax(P_out_advance)
    azimuth_angle_advance = advance_Ang[index]

    for theta2  in range(len(advance_Ang2)):
        kappa = np.array([np.cos(np.pi*advance_Ang2[theta2]/180.0)*np.cos(np.pi*phi/180.0),np.sin(np.pi*advance_Ang2[theta2]/180.0)*np.cos(np.pi*phi/180.0),np.sin(np.pi*phi/180.0)])
        P_out_advance2[theta2] =cost_MUSIC(kappa,MicPos,PN,k)
    index = np.argmax(P_out_advance2)
    azimuth_angle_advance2 = advance_Ang2[index]

    return azimuth_angle_advance,azimuth_angle_advance2


def DOA(MicSignal,Max_delay,MIC_GROUP,tdoa_matrix,tdoa_measures,fs):
    tau = [0] * 5
    len_of_sig= len(MicSignal[0])
    n = len_of_sig*2
    MicSignal = np.fft.rfft(MicSignal, n)

    # estimate each group of delay
    # gcc_phat
    interp = 7
    max_tau = Max_delay
    for i, v in enumerate(MIC_GROUP):
        SIG = MicSignal[v[0]]
        RESIG = MicSignal[v[1]]
        R = SIG * np.conj(RESIG)
        cc = np.fft.irfft(R / np.abs(R), n=(interp * n))
        max_shift = int(interp * n/2 )
        if max_tau:
            max_shift = np.minimum(int(interp * fs * max_tau), max_shift)
        cc = np.concatenate((cc[-max_shift:], cc[:max_shift+1]))
        # find max cross correlation index
        shift = np.argmax(np.abs(cc)) - max_shift
        tau[i] = shift / float(interp * fs)

        # least square solution of (cos, sin)
    sol = np.linalg.pinv(tdoa_matrix).dot( \
            (tdoa_measures * np.array(tau)).reshape(5, 1))
    phi_in_rad = min( sol[0] / math.cos(math.atan2(sol[1],sol[0]) ), 1)

    # phi in degree
    phi = 90 - np.rad2deg( math.asin(phi_in_rad))
    direction = [(math.atan2(sol[1], sol[0])/np.pi*180.0 + 177.0) % 360, phi]

    return direction
############################################################






############################################################
if __name__ == '__main__':
    MicNumber = 6
    radius = 0.047
    select_range= np.arange(256,384,1)
    MicPos = UCAMic(radius,MicNumber)
    P_half = Mix_3D_pro_function_one_source_simulation(MicPos)
    t_start=time.time()



    #k,PN=MUSIC_Parameter(P_half,16000,1,select_range)
    #azimuth_angle,elevation_angle=MUSIC_Grid_Search(MicPos,PN,k)
    #azimuth_angle,elevation_angle=MUSIC_Localization_freqrange_grid_search(P_half,MicPos,1,select_range)
    #azimuth_angle,elevation_angle=MUSIC_Localization_freqrange_theta_constant_phi(P_half,MicPos,1,select_range,20)
    #azimuth_angle,elevation_angle=MUSIC_PSO_localization(P_half,MicPos,1,select_range)
    tend=time.time()
    print(azimuth_angle,elevation_angle,tend-t_start)
    #azimuth,elevation,swarm= MUSIC_PSO_localization(P_half,MicPos,1)
    #azimuth_angle,elevation_angle=MUSIC_Localization(P_half,MicPos,1)
