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
        header=0, 
        names=col_names, 
        usecols=['xa','ya','za','act'])
    if act:
        dat = dat.loc[dat.act == act]

    return dat


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


#times = 0
#for fn in data_files[:10]:
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


def get_spec_peaks(dat, nFFT=128, novr=0, nperseg=128):
    #sp = plt.specgram(dat, NFFT=nFFT, noverlap=novr)
    f,t,Sxx = spectrogram(dat, nfft=nFFT, noverlap=novr, nperseg=nperseg)
    #plt.show()
    pk1 = []
    pk2 = []
    for t in range(Sxx.shape[0]):
        #p_ind = find_peaks(Sxx[:][t])
        p_ind = find_peaks(Sxx[t][:])
        if len(p_ind) >= 2:
            p1, p2 = (dat[p_ind[0]], dat[p_ind[1]])
            pk1.append(p1)
            pk2.append(p2)
    #print pk1
    #print pk2
    #plt.plot(pk1,pk2, '.')
    #plt.show()

    return pk1, pk2


def peaks_for_all(data_files):
    '''calculates spectogram peaks for all files'''
    peak1, peak2, subjn = [], [], []
    for n, fn in enumerate(data_files):
        print os.path.basename(fn)
        dat = pd.read_csv(fn, header=0, names=col_names)
        pk1,pk2 = get_spec_peaks(dat.xa)
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


if 0:
    dat = dat.loc[seg_oi[0][0]:seg_oi[0][1]][:]
    dat = dat.reset_index(drop=True)

    pca = PCA(n_components=2)
    #pca.fit(dat)
    pca.fit(dat[['xa','ya','za']])
    #print "PCA components"
    #print pd.DataFrame(pca.components_.round(2), columns=dat.columns[1:])
    #print "Explained Variance"
    #print pca.explained_variance_ratio_
    #print "Cumulative explained variance"
    #csum = np.cumsum(pca.explained_variance_ratio_)
    #print csum
    #print
    print

    dat2 = pca.transform(dat[xyz])

#data_est = pd.DataFrame(pca2.inverse_transform(dat), columns=data.columns)

if 0:
    n_samp = np.floor(dat.ts[-1:]/.03)
    ts_ = np.arange(0,n_samp*.03,.03)
    ham_win = get_window('hamming', 128)
    print ham_win.shape
    dat1 = resample(dat[xyz], n_samp, window='hamming')
    dat1 = pd.DataFrame(dat1, columns=xyz)
    dat1['ts'] = ts_

    plt.plot(dat.ts, dat.xa)
    plt.plot(dat1.ts, dat1.xa)
    plt.show()