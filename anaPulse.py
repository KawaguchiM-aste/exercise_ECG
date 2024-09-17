import numpy as np
from matplotlib import pyplot as plt
from opensignalsreader import OpenSignalsReader
from biosppy.signals import ecg,bvp
import random

def func_detect_peak(x, t, sw=0, fname_export='resultPeaktimings'):
    t_win = 5
    dt = np.mean(np.diff(t))
    Fs = 1/dt
    out = bvp.bvp(signal=x, sampling_rate=Fs, show=False)
    y = out[1]
    tR = out[2]*dt
    tshow=(np.max(t)-t_win)*random.random()
    itshow=[int(tshow*Fs),int((tshow+t_win)*Fs)]
    plt.figure(figsize=[8,5])
    plt.subplot(2,1,1)
    plt.plot(t,x,'k-',alpha=0.2)
    plt.plot(t,y,'b')
    plt.plot(t[itshow[0]:itshow[1]],y[itshow[0]:itshow[1]],'r')
    plt.ylabel('V_Pulse')
    plt.xlim(np.min(t),np.max(t))
    plt.subplot(2,1,2)
    plt.plot(t,x,'k-',alpha=0.2)
    plt.plot(t,y,'r')
    plt.ylabel('V_ECG')
    for tt in tR:
        i = int(tt*Fs)
        if t[i]>tshow and t[i]<tshow+t_win:
            plt.text(t[i],y[i],'Peak')
    plt.xlim(tshow, tshow+t_win)
    plt.xlabel('Time [s]')
    plt.pause(1)
    if sw==1:
        plt.savefig('{}_P.png'.format(fname_export)) 
    return tR