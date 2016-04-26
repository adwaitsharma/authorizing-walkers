import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.signal import resample
from scipy.signal import get_window
from scipy.signal import find_peaks_cwt, spectrogram
import time


#
# Get started!
#

# column names
col_names = ['ts','xa','ya','za','act']
# xyz motion columns
xyz = ['xa','ya','za']


#
# Helper functions
#

def load_file(file_path, act=None, col_names=col_names):
    '''reads file with appropriate header information'''
    dat = pd.read_csv(file_path, 
        names=col_names, 
        usecols=['xa','ya','za','act'])
    if act:
        dat = dat.loc[dat.act == act]
    
    act_file = file_path[:-4] + '.txt'
    act_dat = pd.read_csv(act_file, names=['act'])
    dat['act'] = act_dat

    dat['mag'] = signal_magnitude(dat)

    return dat

def signal_magnitude(dat):
    mag = np.sqrt(
        (dat.xa - dat.xa.mean())**2 +
        (dat.ya - dat.ya.mean())**2 +
        (dat.za - dat.za.mean())**2)
    return mag


def activity_segs(dat):
    inds = []
    for a in range(1,8):
        mn = min(dat.act[dat.act == a].index)
        mx = max(dat.act[dat.act == a].index)
        inds.append([mn,mx])
    return inds


def get_segs(dat):
    inds =[]
    a = dat.act[0]
    ai = 0
    for r in dat.itertuples():
        b = r.act
        if a != b:
            inds.append([a, ai, r.Index-1])
            a = b
            ai = r.Index
    return inds


def prepare_data():
    '''prepare data with train and test
    each Xy row is a 

    return X_train, y_train, X_test, y_test
    '''
    
    #


def show_fft(dat):
    #dat = pd.read_csv(fn, header=0, names=col_names)
    #print fn
    #dat[xyz].plot()
    #plt.show()
    #t = dat.ts.iloc[-1]
    #print t
    #times +=t
    #print avg_ts(dat.ts)
    dt = 1/52.
    nFFT = 256
    #n_samp = np.int(dat.ts[-1:]/dt)
    n_samp = dat.shape[0]
    #ts_ = np.arange(0,n_samp*dt,dt)
    ts_ = np.linspace(0,(n_samp-1)*dt, num=n_samp)
    

    plt.psd(dat.xa[:1024], nFFT, 1/dt)
    plt.psd(dat.ya[:1024], nFFT, 1/dt)
    plt.psd(dat.za[:1024], nFFT, 1/dt)
    #plt.title(os.path.basename(fn))
    plt.show()


def get_spec_peaks(dat, n_peaks=8, nFFT=128, fs=52, novr=0, nperseg=128):
    '''returns spetrogram peaks for one time series'''

    #sp = plt.specgram(dat, NFFT=nFFT, noverlap=novr)
    f,t,Sxx = spectrogram(dat, fs=fs, nfft=nFFT, noverlap=novr, nperseg=nperseg)
    #plt.show()
    pk1 = []
    pk2 = []
    pks = []
    for t in range(Sxx.shape[0]):
        #p_ind = find_peaks(Sxx[:][t])
        p_ind = find_peaks(Sxx[t][:])
        #print 'p_ind', p_ind
        #print 'Sxx[x][:]', len(Sxx[t])
        if len(p_ind) >= 2:
            #p1, p2 = (dat.iloc[p_ind[0]], dat.iloc[p_ind[1]])
            p1, p2 = (f[p_ind[0]], f[p_ind[1]])
            pk1.append(p1)
            pk2.append(p2)
        #print len(p_ind), len(f)
        
        cur_peak_inds = p_ind[:n_peaks]
        #while len(cur_peak_inds) < n_peaks:
        #    cur_peak_inds.append(0)
        #    print '.',
        #print
        pks.append([f[i] for i in p_ind[:n_peaks]])
        #print len(pks[-1])

    #print pk1
    #print pk2
    #plt.plot(pk1,pk2, '.')
    #plt.show()

    #return pk1, pk2
    pks = np.array(pks)
    return pks

def plt_harmon(dat, nFFT=128, fs=52, novr=0, nperseg=128):
    plt.figure()
    

    if len(dat.shape) > 1:
        for i, c in zip(['xa', 'ya', 'za'], ['r','g','b']):
            p1, p2 = get_spec_peaks(dat[i], nFFT=128, fs=52, novr=0, nperseg=128)
            plt.scatter(p1,p2, color=c)
        #plt.show()
    else:
        p1, p2 = get_spec_peaks(dat, nFFT=128, fs=52, novr=0, nperseg=128)
        mean_p1 = np.mean(p1)
        mean_p2 = np.mean(p2)
        plt.scatter(p1,p2, color='b')
        plt.plot(mean_p1, mean_p2, 'ro')
        #plt.show()


def peaks_for_all(data_files):
    '''calculates spectogram peaks for all files'''
    peak1, peak2, subjn = [], [], []
    for n, fn in enumerate(data_files):
        print os.path.basename(fn)
        #dat = pd.read_csv(fn, header=0, names=col_names)
        dat = load_file(fn, act=4)
        pk1,pk2 = get_spec_peaks(dat.mag)
        subj = [n] * len(pk1)
        peak1.append(pk1)
        peak2.append(pk2)
        subjn.append(subj)
    results = pd.DataFrame({'p1':peak1, 'p2':peak2, 'sub':subjn})
    #plt.figure()
    #plt.plot(peak1,peak2, '.', color=subjn)
    #plt.show()
    return results


def plot_peaks(res):
    n = len(res)

    for i in range(n):
        plt.plot(res['p1'][i], res['p2'][i], '.')

    plt.show()



def find_peaks(dat):
    pks = np.arange(1,5)
    p_ind = find_peaks_cwt(dat, pks)

    return p_ind

def adj_segs(segs, x):
    segs[0][2] += x
    for i in range(1, len(segs)-1):
        segs[i][1] += x
        segs[i][2] += x
    segs[-1][1] += x
    return segs




def pltsegs(segs, fs=52., yd=[0,1]):
    for i,l in enumerate(segs):
        #xd = [l[1]/fs,l[1]/fs]
        xd = [l[2]/fs, l[2]/fs]
        yd = [0,fs/2]
        #print xd
        plt.plot(xd, yd, linewidth=2)
        #plt.text(xd[0],yd[1], str(i+1))
        plt.text(xd[0],yd[1], str(l[0]))


def spec_3a(dat, segs=None, nFFT=128, novr=0, fs=52.):   
    if segs is None:
        segs = get_segs(dat)

    plt.figure(figsize=(20,6))
    plt.subplot(311)
    plt.specgram(dat.xa, Fs=fs, NFFT=nFFT, noverlap=novr)
    pltsegs(segs)
    plt.subplot(312)
    plt.specgram(dat.ya, Fs=fs, NFFT=nFFT, noverlap=novr)
    pltsegs(segs)
    plt.subplot(313)
    plt.specgram(dat.za, Fs=fs, NFFT=nFFT, noverlap=novr)
    pltsegs(segs)
    plt.show()


def acc_3a(dat):
    segs = get_segs(dat)
    ns = dat.shape[0] 
    ts = np.linspace(0, ns/52., num=ns)

    plt.figure(figsize=(20,6))
    plt.subplot(311)
    plt.plot(ts, dat.xa)
    pltsegs(segs)
    plt.subplot(312)
    plt.plot(ts, dat.ya)
    pltsegs(segs)
    plt.subplot(313)
    plt.plot(ts, dat.za)
    pltsegs(segs)
    plt.show()



def analysis(data_files):
    results = peaks_for_all(data_files)
    plot_peaks(results)

