import numpy as np
import math

from numpy.linalg import inv
from numpy.linalg import norm
import math
from numpy.linalg import solve, norm
from scipy.io.wavfile import read,write
from snr_seg_one import snr_seg_one
import pdb
from matplotlib import pyplot
from scipy.signal import find_peaks

from matplotlib import pyplot as plt

def tra_process_new(y):

    y_len                 = len(y)
# sig_len               = 208896;
# length_add            = sig_len - y_len;
# y1_extend(length_add) = 0;
    hann_len              = hann_len
    frame_number          = 2*math.floor(y_len/hann_len)-1#%(sig_len - hann_len)/(0.5*hann_len) + 1;


    signal_seg=np.zeros([frame_number,hann_len])
    y1_extend_sig         = y.transpose() #[y',y1_extend];


    #signal_seg(frame_number,hann_len) = 0;
    w_nn                  = np.hanning(hann_len+1)#hann(hann_len);+
    w_nn                  = w_nn[:hann_len]
    n_fft                 = 1*hann_len

    u0 = 1
    s = 1

    #H_W(0.5*n_fft+1) = 0;
    H_W = np.zeros([1,int(0.5*n_fft+1)])

    #sigma_d(0.5*n_fft+1,frame_number) = 0;
    sigma_d = np.zeros([int(0.5*n_fft+1),frame_number])

    #alpha_d(0.5*n_fft+1,frame_number) = 0;
    alpha_d = np.zeros([int(0.5*n_fft+1),frame_number])

    #gamma_d(0.5*n_fft+1,frame_number) = 0;
    gamma_d= np.zeros([int(0.5*n_fft+1),frame_number])

    beta_d = 0.6

    #gamma_dy(0.5*n_fft+1,frame_number) = 0;
    gamma_dy = np.zeros([int(0.5*n_fft+1),frame_number])
    #gamma_dy_val = np.zeros([1,int(0.5*n_fft+1)])

    #seg_snr_vector(frame_number) = 0;
    seg_snr_vector = np.zeros([1,frame_number])

    #PSN_predict(0.5*n_fft+1,frame_number) = 0;
    PSN_predict = np.zeros([int(0.5*n_fft+1),frame_number])

    #S_filter_full(hann_len)            = 0;
    S_filter_full = np.zeros([1,hann_len],dtype = complex)
    signal_syn = np.zeros([1,y_len])
    mean_sigma_d = np.zeros([1,int(0.5*n_fft+1)])
    yd_ratio     = np.zeros([1,int(0.5*n_fft+1)])
    alpha_denominator = np.zeros([1,int(0.5*n_fft+1)])
    for segment in range(frame_number):#segment = 1:frame_number

      #    len_extend(hann_len) = 0;
      #    len_extend = np.zeros([1,hann_len])

      # noisy speech
      #    y1_w = 0.5.*y1_extend_sig(1+(segment-1)*hann_len*0.5:hann_len+(segment-1)*hann_len*0.5).* w_nn.transpose();
          y1_w = 0.5*np.multiply(y1_extend_sig[int(0+(segment)*hann_len*0.5):int(hann_len+(segment)*hann_len*0.5)], w_nn.transpose())
  #    c = np.dot(a,b)
  #    c = np.multiply(a,b)
          y1_w_e = y1_w#%[y1_w,len_extend];
          #y1_freq = fft(y1_w_e)/n_fft;
          y1_freq = np.fft.fft(y1_w_e)/n_fft
         # y1_freq = y1_freq(1:0.5*n_fft+1);
          y1_freq = y1_freq[0:int(0.5*n_fft+1)]
         # PSN = (abs(y1_freq).^2);
          PSN = np.square(abs(y1_freq))


          if segment <= 9:
            #  sigma_d(:,segment) = PSN;
            #  gamma_d(:,segment) = 0;
            #  gamma_dy(:,segment) = 0.001;
              sigma_d[:,segment] = PSN
              gamma_d[:,segment] = 0
              gamma_dy[:,segment] = 0.001
          else:
              for index2 in range(int(0.5*n_fft+1)):#index2 = 1:0.5*n_fft+1
               #   mean_sigma_d(index2) = 0.1 * (sigma_d(index2,segment-1) + sigma_d(index2,segment-2) + sigma_d(index2,segment-3) + sigma_d(index2,segment-4)+...
               #                sigma_d(index2,segment-5) + sigma_d(index2,segment-6) + sigma_d(index2,segment-7) + sigma_d(index2,segment-8) + sigma_d(index2,segment-9)+...
               #                sigma_d(index2,segment-10));
                  mean_sigma_d[0,index2] = 0.1 * (sigma_d[index2,segment-1] + sigma_d[index2,segment-2] + sigma_d[index2,segment-3] + sigma_d[index2,segment-4]+\
                               sigma_d[index2,segment-5] + sigma_d[index2,segment-6] + sigma_d[index2,segment-7] + sigma_d[index2,segment-8] + sigma_d[index2,segment-9]+\
                               sigma_d[index2,segment-10])
         #     end
         # end
         # %mean_sigma_d = 0.1*sum_sigma_d;
              #pdb.set_trace()
              for index in range(int(0.5*n_fft+1)):#index = 1:0.5*n_fft+1
                  if segment == 0:#segment == 1
                  #PSN_predict(index,segment) = PSN(index);
                      PSN_predict[index,segment] = PSN[index]
                  else:
                #  PSN_predict(index,segment) = 0.5 .* PSN_predict(index,segment-1) + 0.5 .* PSN(index);
                      PSN_predict[index,segment] = 0.5 * PSN_predict[index,segment] + 0.5 * PSN[index]
             # end

          #end

          #gamma_d(:,segment) = PSN ./ mean_sigma_d;
              gamma_d[:,segment] = PSN / mean_sigma_d

          #alpha_denominator = 1 + (2.718).^ (-beta_d * (gamma_d(:,segment)-1.5));
              for index in range(int(0.5*n_fft+1)):
                   alpha_denominator[0,index] = 1 + math.pow(2.718,(-beta_d * (gamma_d[index,segment]-1.5)))#(2.718).^ (-beta_d * (gamma_d[:,segment]-1.5));

          #alpha_d(:,segment) = 1 ./ alpha_denominator.transpose();

              alpha_d[:,segment] = 1 / alpha_denominator

          #sigma_d(:,segment) = alpha_d(:,segment) .* sigma_d(:,segment-1) + (1-alpha_d(:,segment)) .* PSN.transpose();
              sigma_d[:,segment] = np.multiply(alpha_d[:,segment],sigma_d[:,segment-1])+np.multiply((1-alpha_d[:,segment]),PSN.transpose())#alpha_d[:,segment] .* sigma_d[:,segment-1] + (1-alpha_d[:,segment]) .* PSN.transpose();
              #pdb.set_trace()
              for index in range(int(0.5*n_fft+1)):#index = 1:0.5*n_fft+1
                   if sigma_d[index,segment] > 0:#sigma_d(index,segment) > 0#%
                 # yd_ratio(index) = PSN_predict(index,segment) ./ sigma_d(index,segment).transpose();
                      yd_ratio[0,index] = PSN_predict[index,segment] / sigma_d[index,segment].transpose()
                   else:
                 # yd_ratio(index) = 1000;
                      yd_ratio[0,index] = 1000
             # end
         # end

         #   gamma_dy(:,segment) = (0.5).* yd_ratio(:);
              gamma_dy[:,segment] = (0.5) * yd_ratio[0,:]
     # end
           #   pdb.set_trace()
          #gamma_dy_val[1,:] = gamma_dy[:,segment]
          for index in range(int(0.5*n_fft+1)): #index = 1:0.5*n_fft+1

                if segment <= 9:
                #H_W(index) = gamma_dy(index,segment) ./ (10 + gamma_dy(index,segment));
                    #pdb.set_trace()
                    H_W[0,index] = gamma_dy[index,segment] / (10 + gamma_dy[index,segment])
                    #pdb.set_trace()
                    #H_W[1,index] = gamma_dy_val[1,index] / (10 + gamma_dy_val[1,index])
                    #pdb.set_trace()
                else:
                    um = u0 - (seg_snr_vector[0,segment-1] / s)

                    if um >= 0.01:
                        u = um
                    else:
                        u = 0.01
                    #end
                    #H_W(index) = gamma_dy(index,segment) ./ (1*u + gamma_dy(index,segment));
                    H_W[0,index] = gamma_dy[index,segment] / (1*u + gamma_dy[index,segment])
                    #H_W[1,index] = gamma_dy_val[1,index] / (1*u + gamma_dy_val[1,index])
                #end
         #end


        #S_filter = y1_freq .* H_W;
          S_filter = np.multiply(y1_freq , H_W)


        #a_middle = real(S_filter(0.5*n_fft+1));
          a_middle = S_filter[0,int(0.5*n_fft)].real

        #S_filter_full(0.5*n_fft+1) = a_middle;
          S_filter_full[0,int(0.5*n_fft)] = a_middle


        # in order to get real number after IFFT, we do inverse order.
          for index in range(int(0.5*n_fft)):#index = 1:0.5*n_fft
          #S_filter_full(index) = (S_filter(index));
              S_filter_full[0,index] = (S_filter[0,index])
        #end



          for index in range(int(0.5*n_fft-1)):#index = 0:0.5*n_fft-2
         # S_filter_full(hann_len-index) = conj(S_filter_full(index+2));
              S_filter_full[0,hann_len-index-1] = np.conj(S_filter_full[0,index+1])


        #end
        #s_filter = ifft(S_filter_full);

          s_test   =  S_filter_full
          s_filter_o = np.fft.ifft(s_test)


        #signal_seg(segment,1:hann_len) = s_filter(1:hann_len);
          signal_seg[segment,0:hann_len] = s_filter_o[0:hann_len]
        #segSNR = snr_seg_one(s_filter(1:hann_len), y1_w, hann_len);
          segSNR = snr_seg_one(s_filter_o[0:hann_len].T, y1_w, hann_len)
        #seg_snr_vector(segment) = segSNR;
          seg_snr_vector[0,segment] = segSNR



#      end

# signal synthesis
    #signal_syn(y_len) = 0;
    #pdb.set_trace()
    for index_n in range(frame_number):#index_n = 1:frame_number
        #signal_syn(1+(index_n-1)*hann_len*0.5:hann_len+(index_n-1)*hann_len*0.5) = signal_syn(1+(index_n-1)*hann_len*0.5:hann_len+(index_n-1)*hann_len*0.5) + signal_seg(index_n,1:hann_len);
        signal_syn[0,int(0+(index_n)*hann_len*0.5):int(hann_len+(index_n)*hann_len*0.5)] = signal_syn[0,int(0+(index_n)*hann_len*0.5):int(hann_len+(index_n)*hann_len*0.5)] + signal_seg[index_n,0:hann_len]
    #end
    max_val = np.amax(abs(signal_syn))
    print(max_val)
    #pdb.set_trace()
    max_val = np.max(abs(signal_syn))
    print(max_val)
    signal_syn = signal_syn / max_val;
    #max_val_s  =  np.max(abs(signal_syn)
    #pdb.set_trace()
    #signal_syn = signal_syn / max_val_s
    xo = signal_syn
 #   pdb.set_trace()
    print("done")
    return xo

def UCAMic(radius,MicNumber):
    mic_theta           = np.arange(0, 2*np.pi, 2*np.pi/MicNumber)
    MicPos = radius*np.array([np.cos(mic_theta),np.sin(mic_theta),np.zeros(6)])
    return MicPos

def Mix_3D_pro_function_one_source_simulation(MicPos):
    ## transpose error ? +- for image part in weight
    Ts     = 2048/16000
    c      = 343.0
    fs     = 16000
    SorNum = 1
    SorLen = int(fs*Ts)
    sr,x1  = read('C:/Users/User/Desktop/DARPA/male_16k.wav')
    x1     = x1/32767.0
    x1     = x1[999:999+SorLen]
    m,n    = MicPos.shape
    source =np.reshape(x1,[1,SorLen])
    SorPos = np.zeros((SorNum,2))
    kappa  = np.zeros((SorNum,3))
    for i in range(SorNum):
        theta       = 181
        phi         = 60
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

def MUSIC_PSO_localization(P_half,MicPos,SorNum):
    c      = 343.0
    fs     = 16000
    NWIN       = int(2048)
    win        = np.hanning(NWIN+1)
    win        = win[:NWIN]
    ## FFT
    NFFT       = int(2048)
    df         = fs/NFFT
    Freqs      = np.arange(0,(NFFT/2)*df,df)
    inertia=0.4
    correction_factor=1.2
    correction_factor_group=1.6
    Rshare=40
    iterations=20
    swarm =np.zeros((22,7))
    Ang = np.arange(30,360,30)
    Yang = np.arange(30,90,30)
    counter=0
    Distance_matrix = np.zeros((22,22))
    M=np.zeros((22,22))
    P_out = np.zeros(128)
    temp1=np.zeros(2)
    k = 2*np.pi*Freqs/c
    for x in range(11):
        for y in range(2):
            swarm[counter,0:2]=np.array([Ang[x],Yang[y]])
            counter = counter + 1

    for iter  in range(iterations):
        for i in range(22):
            for ScanFrequency in range(256,384,1):
            #print(ScanFrequency)
                x_1 = P_half[:,select_range[ScanFrequency],:]
                Rxx =np.dot(x_1,np.conj(x_1.T))
                a, b = np.linalg.eig(Rxx)
                US = b[:,a.argsort()[::-1]][:,:SorNum]
                PN = np.identity(6)-np.dot(US,np.conj(US.T))
                kappa = np.array([np.cos(np.pi*swarm[i,0]/180.0)*np.cos(np.pi*swarm[i,1]/180.0),np.sin(np.pi*swarm[i,0]/180.0)*np.cos(np.pi*swarm[i,1]/180.0),np.sin(np.pi*swarm[i,1]/180.0)])
                SteeringVector =np.reshape(np.exp(1j*k[ScanFrequency]*np.dot(kappa,MicPos)),[6,1])
                P_out[ScanFrequency-64] =np.abs(1/np.dot(np.dot(np.conj(SteeringVector.T),PN),SteeringVector))
            P_sum = np.sum(P_out)
            if P_sum > swarm[i,6]:
                for q in range(2):
                    swarm[i,q+2] = swarm[i,q]
            swarm[i,6]= P_sum
            for j in range(i,22):
                if j==i:
                    Distance_matrix[i,j]= 0
                else:
                    Distance_matrix[i,j] = norm(swarm[i,0:2]-swarm[j,0:2])

                Distance_matrix[j,i]=Distance_matrix[i,j]

            for z  in range(22):
                if Distance_matrix[i,z]<Rshare:
                    M[z,i]=1
                else:
                    M[z,i]=0

        Follow_matrix = np.zeros((22,22))

        for i in range(22):
            Decision_matrix = np.multiply(swarm[:,6],M[:,i])
            fbest = np.argmax(Decision_matrix)
            Follow_matrix[i,fbest] = 1
            for q in range(2):
                swarm[i,q+4] = inertia*swarm[i,q+4]+correction_factor*np.random.rand()*(swarm[i,q+2]-swarm[i,q])+correction_factor_group*np.random.rand()*(swarm[fbest,q+2]-swarm[i,q])
            for j in range(2):
                swarm[i,j] =swarm[i,j] + 0.8*swarm[i,j+4]

        best_swarm_index = np.argmax(np.sum(Follow_matrix,axis=0))
        best_swarm = swarm[best_swarm_index,0:2]
        if iter<=1:
            temp1 =best_swarm
        else:
            if norm(best_swarm-temp1) <1:
                #print(iter)
                break
            else:
                temp1=best_swarm

    return best_swarm[0],best_swarm[1]


def MUSIC_Localization(P_half,MicPos,SorNum):
    c      = 343.0
    fs     = 16000
    NWIN       = int(2048)
    win        = np.hanning(NWIN+1)
    win        = win[:NWIN]
    ## FFT
    NFFT       = int(2048)
    df         = fs/NFFT
    Freqs      = np.arange(0,(NFFT/2)*df,df)
    Ang = np.arange(0,360,30)
    Yang = np.arange(0,90,30)
    P_out=np.zeros((128,3,12))
    MicNumber,NumofFreqs,NumofFrame = P_half.shape
    h =np.transpose(P_half,[1,0,2])
    for ScanFrequency in range(64,192,1):
        #print(ScanFrequency)
        k = np.divide(2*np.pi*Freqs[ScanFrequency],c)
        x_1 = P_half[:,select_range[ScanFrequency],:]
        Rxx =np.dot(x_1,np.conj(x_1.T))
        a, b = np.linalg.eig(Rxx)
        US = b[:,a.argsort()[::-1]][:,:SorNum]
        PN = np.identity(6)-np.dot(US,np.conj(US.T))

        for theta  in range(len(Ang)):
            for phi in range(len(Yang)):
                kappa = np.array([np.cos(np.pi*Ang[theta]/180.0)*np.cos(np.pi*Yang[phi]/180.0),np.sin(np.pi*Ang[theta]/180.0)*np.cos(np.pi*Yang[phi]/180.0),np.sin(np.pi*Yang[phi]/180.0)])
                SteeringVector =np.reshape(np.exp(1j*k*np.dot(kappa,MicPos)),[6,1])
                P_out[ScanFrequency-64,phi,theta] =np.abs(1/np.dot(np.dot(np.conj(SteeringVector.T),PN),SteeringVector))

    P_mean = np.mean(P_out,axis = 0)
    MaxValue = np.max(np.max(P_mean,axis=1),axis=0)
    location = np.argwhere(P_mean==MaxValue)

    azimuth_angle = Ang[location[0,1]]
    elevation_angle =Yang[location[0,0]]
    return azimuth_angle,elevation_angle


def MUSIC_Localization_freqrange(P_half,MicPos,SorNum,select_range):
    c      = 343.0
    fs     = 16000
    NWIN       = int(2048)
    ## FFT
    NFFT       = int(2048)
    df         = fs/NFFT
    Freqs      = np.arange(0,(NFFT/2)*df,df)
    length_select_range = len(select_range)
    Ang = np.arange(0,360,5)
    Yang = np.arange(0,-90,-5)
    P_out=np.zeros((length_select_range,18,72))


    for ScanFrequency in range(length_select_range):
        #print(ScanFrequency)
        k = np.divide(2*np.pi*Freqs[select_range[ScanFrequency]],c)
        x_1 = P_half[:,select_range[ScanFrequency],:]
        Rxx =np.dot(x_1,np.conj(x_1.T))
        #Rxx = np.dot(P_half[:,select_range[ScanFrequency]].reshape((6,1)),np.conj(P_half[:,select_range[ScanFrequency]].reshape((1,6))))
        a, b = np.linalg.eig(Rxx)
        #a_sorted = np.argsort(a)[::-1]
        US = b[:,a.argsort()[::-1]][:,:SorNum]
        PN = np.identity(6)-np.dot(US,np.conj(US.T))

        for theta  in range(len(Ang)):
            for phi in range(len(Yang)):
                kappa = np.array([np.cos(np.pi*Ang[theta]/180.0)*np.cos(np.pi*Yang[phi]/180.0),np.sin(np.pi*Ang[theta]/180.0)*np.cos(np.pi*Yang[phi]/180.0),np.sin(np.pi*Yang[phi]/180.0)])
                SteeringVector =np.reshape(np.exp(1j*k*np.dot(kappa,MicPos)),[6,1])
                P_out[ScanFrequency,phi,theta] =np.abs(1/np.dot(np.dot(np.conj(SteeringVector.T),PN),SteeringVector))

    P_mean = np.mean(P_out,axis = 0)
    MaxValue = np.max(np.max(P_mean,axis=1),axis=0)
    # print("P_mean size: ", P_mean.shape)
    # print("Max size: ", MaxValue.shape)
    # cax = plt.matshow(P_mean, interpolation='nearest')
    # plt.colorbar(cax)
    # plt.ylabel('elevation')
    # plt.xlabel('azimuth')
    # plt.xticks(np.arange(0, P_mean.shape[1], step=5), np.arange(0, 360, step=25))
    # plt.yticks(np.arange(0, P_mean.shape[0], step=2), np.arange(0, -90, step=10))
    # plt.show()
    location = np.argwhere(P_mean==MaxValue)
    azimuth_angle = Ang[location[0][1]]
    elevation_angle =Yang[location[0][0]]
    return azimuth_angle,elevation_angle

def MUSIC_Localization_freqrange_theta(P_half,MicPos,SorNum,select_range):
    c      = 343.0
    fs     = 16000
    NWIN       = int(2048)
    ## FFT
    NFFT       = int(2048)
    df         = fs/NFFT
    Freqs      = np.arange(0,(NFFT/2)*df,df)
    length_select_range = len(select_range)
    Ang = np.arange(0,360,5)
    P_out=np.zeros((length_select_range,72))
    # phi = 20
    phi = 40

    for ScanFrequency in range(length_select_range):
        #print(ScanFrequency)
        k = np.divide(2*np.pi*Freqs[select_range[ScanFrequency]],c)
        x_1 = P_half[:,select_range[ScanFrequency],:]
        Rxx =np.dot(x_1,np.conj(x_1.T))
        #Rxx = np.dot(P_half[:,select_range[ScanFrequency]].reshape((6,1)),np.conj(P_half[:,select_range[ScanFrequency]].reshape((1,6))))
        a, b = np.linalg.eig(Rxx)
        #a_sorted = np.argsort(a)[::-1]
        US = b[:,a.argsort()[::-1]][:,:SorNum]
        PN = np.identity(6)-np.dot(US,np.conj(US.T))

        for theta  in range(len(Ang)):
            kappa = np.array([np.cos(np.pi*Ang[theta]/180.0)*np.cos(np.pi*20/180.0),np.sin(np.pi*Ang[theta]/180.0)*np.cos(np.pi*20/180.0),np.sin(np.pi*20/180.0)])
            SteeringVector =np.reshape(np.exp(1j*k*np.dot(kappa,MicPos)),[6,1])
            P_out[ScanFrequency,theta] =np.abs(1/np.dot(np.dot(np.conj(SteeringVector.T),PN),SteeringVector))

    P_mean = np.mean(P_out,axis = 0)
    index = np.argmax(P_mean)
    azimuth_angle = Ang[index]

    return azimuth_angle,phi




def Multi_MUSIC_Localization_freqrange_theta(P_half,MicPos,SorNum,select_range):
    c      = 343.0
    fs     = 16000
    NWIN       = int(2048)
    ## FFT
    NFFT       = int(2048)
    df         = fs/NFFT
    Freqs      = np.arange(0,(NFFT/2)*df,df)
    length_select_range = len(select_range)
    Ang = np.arange(0,360,5)
    P_out=np.zeros((length_select_range,72))
    phi = 20

    for ScanFrequency in range(length_select_range):
        #print(ScanFrequency)
        k = np.divide(2*np.pi*Freqs[select_range[ScanFrequency]],c)
        x_1 = P_half[:,select_range[ScanFrequency],:]
        Rxx =np.dot(x_1,np.conj(x_1.T))
        #Rxx = np.dot(P_half[:,select_range[ScanFrequency]].reshape((6,1)),np.conj(P_half[:,select_range[ScanFrequency]].reshape((1,6))))
        a, b = np.linalg.eig(Rxx)
        #a_sorted = np.argsort(a)[::-1]
        US = b[:,a.argsort()[::-1]][:,:SorNum]
        PN = np.identity(6)-np.dot(US,np.conj(US.T))

        for theta  in range(len(Ang)):
            kappa = np.array([np.cos(np.pi*Ang[theta]/180.0)*np.cos(np.pi*20/180.0),np.sin(np.pi*Ang[theta]/180.0)*np.cos(np.pi*20/180.0),np.sin(np.pi*20/180.0)])
            SteeringVector =np.reshape(np.exp(1j*k*np.dot(kappa,MicPos)),[6,1])
            P_out[ScanFrequency,theta] =np.abs(1/np.dot(np.dot(np.conj(SteeringVector.T),PN),SteeringVector))
    P_mean = np.mean(P_out,axis = 0)
    peaks , _ =find_peaks(P_mean)
    index = peaks[np.argsort(P_mean[peaks])][-SorNum:]
    azimuth_angle = Ang[index[0]]
    azimuth_angle2 = Ang[index[2]]

    return azimuth_angle,azimuth_angle2



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

if __name__ == '__main__':
    MicNumber = 6
    radius = 0.05
    select_range= np.arange(256,384,1)
    MicPos = UCAMic(radius,MicNumber)
    P_half = Mix_3D_pro_function_one_source_simulation(MicPos)
    azimuth_angle,elevation_angle=MUSIC_Localization_freqrange(P_half,MicPos,1,select_range)
    #azimuth,elevation,swarm= MUSIC_PSO_localization(P_half,MicPos,1)
    #azimuth_angle,elevation_angle=MUSIC_Localization(P_half,MicPos,1)
