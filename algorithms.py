import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.learning_curve import learning_curve
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn import grid_search

from sklearn.metrics import f1_score
from scipy.signal import get_window
from scipy.signal import spectrogram #, find_peaks_cwt

#for peakdet
from numpy import NaN, Inf, arange, isscalar, asarray, array



from time_series_segmentation import peak_detection
import itertools

col_names = ['ts','xa','ya','za','act']

data_dir = os.path.realpath('.') +'\data'
# filenames are 1 to 15
data_files = [data_dir+os.path.sep+str(i)+'.csv' for i in range(1,16)]


#https://gist.github.com/sixtenbe/1178136#file-peakdetect-py
def peakdet(v, delta, x = None):
    """
    Converted from MATLAB script at http://billauer.co.il/peakdet.html
    
    Returns two arrays
    
    function [maxtab, mintab]=peakdet(v, delta, x)
    %PEAKDET Detect peaks in a vector
    %        [MAXTAB, MINTAB] = PEAKDET(V, DELTA) finds the local
    %        maxima and minima ("peaks") in the vector V.
    %        MAXTAB and MINTAB consists of two columns. Column 1
    %        contains indices in V, and column 2 the found values.
    %      
    %        With [MAXTAB, MINTAB] = PEAKDET(V, DELTA, X) the indices
    %        in MAXTAB and MINTAB are replaced with the corresponding
    %        X-values.
    %
    %        A point is considered a maximum peak if it has the maximal
    %        value, and was preceded (to the left) by a value lower by
    %        DELTA.
    
    % Eli Billauer, 3.4.05 (Explicitly not copyrighted).
    % This function is released to the public domain; Any use is allowed.
    
    """
    maxtab = []
    mintab = []
       
    if x is None:
        x = arange(len(v))
    
    v = asarray(v)
    
    if len(v) != len(x):
        sys.exit('Input vectors v and x must have same length')
    
    if not isscalar(delta):
        sys.exit('Input argument delta must be a scalar')
    
    if delta <= 0:
        sys.exit('Input argument delta must be positive')
    
    mn, mx = Inf, -Inf
    mnpos, mxpos = NaN, NaN
    
    lookformax = True
    
    for i in arange(len(v)):
        this = v[i]
        if this > mx:
            mx = this
            mxpos = x[i]
        if this < mn:
            mn = this
            mnpos = x[i]
        
        if lookformax:
            if this < mx-delta:
                maxtab.append((mxpos, mx))
                mn = this
                mnpos = x[i]
                lookformax = False
        else:
            if this > mn+delta:
                mintab.append((mnpos, mn))
                mx = this
                mxpos = x[i]
                lookformax = True

    return array(maxtab), array(mintab)



#
# Helper functions
#

def load_file(file_path, act=None, col_names=col_names):
    '''reads file with appropriate header information'''
    dat = pd.read_csv(file_path, 
        names=col_names, 
        usecols=['xa','ya','za','act'])
    act_file = file_path[:-4] + '.txt'
    act_dat = pd.read_csv(act_file, names=['act'])
    dat['act'] = act_dat
    
    if act:
        dat = dat.loc[dat.act == act]
    
    ts = np.arange(0, dat.shape[0]/52., 1/52.)
    dat['ts'] = ts

    dat['mag'] = signal_magnitude(dat)

    return dat


def load_data(data_files, subjs=range(1,16), act=None, col_names=['ya']):
    subject_number = lambda x: int(os.path.basename(x)[:-4])
    data_files_selected = [i for i in data_files if subject_number(i) in subjs]
    
    data = pd.DataFrame()
    for i, f in enumerate(data_files_selected):
        print subject_number(f),
        if act:
            d = load_file(f, act=act)
        else:
            d = load_file(f)
        subj_col = [subject_number(f)] * d.shape[0]
        d['subj'] = subj_col
        data = data.append(d, ignore_index=True)

    return data


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


def get_activity_segments(dat):
    '''Takes a pandas DataFrame and returns the row numbers where the "act"(ivity)
    column changes values as a tuple of:
    (action number, starting row, ending row)'''

    segment_list = []

    # initial activity
    ind = 0
    old_act = dat.act[ind]
    
    # itterate through rows with named columns
    for r in dat.itertuples():

        new_act = r.act
        if old_act != new_act:
            segment_list.append((old_act, ind, r.Index-1))
            old_act = new_act
            ind = r.Index
    return segment_list


def prepare_data(data_files, dim='xa', n_peaks=5, test_ratio=.3, 
    random_state=3):
    '''prepare data with train and test
    each Xy row is a 

    return X_train, y_train, X_test, y_test
    '''
    #print data_files

    X_train, y_train, X_test, y_test = [], [], [], []

    for i, f in enumerate(data_files):
        print os.path.basename(f)
        dat = load_file(f, act=4)
        X = get_spec_peaks(dat[dim], n_peaks)
        n_samp = len(X)
        y = [i+1] * n_samp

        X_train_i, X_test_i, y_train_i, y_test_i = train_test_split(
            X, y, test_size=test_ratio, random_state=random_state)
        
        X_train.extend(X_train_i)
        X_test.extend(X_test_i)
        y_train.extend(y_train_i)
        y_test.extend(y_test_i)

    X_train, y_train = np.array(X_train), np.array(y_train)
    X_test, y_test = np.array(X_test), np.array(y_test)

    return X_train, y_train, X_test, y_test 


def show_fft(dat, nFFT=256):
    #dat = pd.read_csv(fn, header=0, names=col_names)
    #print fn
    #dat[xyz].plot()
    #plt.show()
    #t = dat.ts.iloc[-1]
    #print t
    #times +=t
    #print avg_ts(dat.ts)
    dt = 1/52.
    #nFFT = 256
    #n_samp = np.int(dat.ts[-1:]/dt)
    n_samp = dat.shape[0]
    #ts_ = np.arange(0,n_samp*dt,dt)
    ts_ = np.linspace(0,(n_samp-1)*dt, num=n_samp)
    

    plt.psd(dat.xa[:1024], nFFT, 1/dt)
    plt.psd(dat.ya[:1024], nFFT, 1/dt)
    plt.psd(dat.za[:1024], nFFT, 1/dt)
    #plt.title(os.path.basename(fn))
    plt.show()


#def plt_walking_psd(data, sig, n_peaks=6, nFFT=256, pk_dist=.3, show_peaks=False):
def plt_walking_psd(data, sig, n_peaks=6, nFFT=256, delta=10, show_peaks=False):
    plt.figure()
    for si in list(set(data.subj)):
        di = data[data.subj == si]
        di = di[di.act == 4]
        x = di[sig][:]
        a,b,l = plt.psd(x, nFFT, 52., color='k', return_line=True)
        # grab data from the plot
        pwr, fq = l[0].get_ydata(), l[0].get_xdata()
        #p = peak_detection(pwr, n_peaks, 0, .02, int(pk_dist*52))
        #print pwr, fq  
        mx, mn = peakdet(pwr, delta=delta)
        mx, mn = mx.astype(int), mn.astype(int)

 
        if show_peaks:
            #for pi in p:
            #print pi
            #plt.plot(fq[pi[0]], pi[1], 'ro')
            for p in mx:
                plt.plot(fq[p[0]], p[1], 'go')
            for p in mn:
                plt.plot(fq[p[0]], p[1], 'ro')

    plt.tight_layout()
    plt.show()
    #return pwr, fq

def plt_psd_w_peaks(x, delta=10):
    #p_ind = find_peaks(x)
    mx, mn = peakdet(x, delta=delta)
    #print mx
    #print mn
    plt.plot(x)
    for p in mx:
        plt.plot(p[0], p[1], 'go')
    for p in mn:
        plt.plot(p[0], p[1], 'ro')
    plt.show()


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


def get_spec_features(Dat, sig_comps='mag', nFFT=256, n_peaks=3, delta=50):
    fs = 52.
    novr = 0
    nperseg = nFFT
    
    if type(sig_comps) is str:
        sig_comps = [sig_comps]
    #pk1,pk2 = get_spec_peaks(Dat.mag, novr=0)

    feats = pd.DataFrame()
    for sig_comp in sig_comps:
        # Calculate the spectrogram
        f,t,Sxx = spectrogram(Dat[sig_comp], fs=fs, nfft=nFFT, noverlap=novr, nperseg=nperseg)
        #print 'Sxx.shape', Sxx.shape
        n_nows = len(t)
        # get activity labels
        inds = np.arange(nFFT/2., nFFT*t.shape[0], nFFT)
        inds = [int(i) for i in inds]
    
        # initialize for number of peaks to return
        peaks = np.zeros((len(t), n_peaks))
        # Find peaks for each time slice
        for ti in range(t.shape[0]):
            #print ti
            mx, mn = peakdet(Sxx[:,ti], delta=delta)
            mx, mn = mx.astype(int), mn.astype(int)
            pk_inds = [i[0] for i in mx]
            #print pk_inds
            pk_freqs = [f[i] for i in pk_inds]
            #print pk_freqs
            if len(pk_freqs) >= n_peaks:
                peaks[ti,:] = pk_freqs[:n_peaks]
            else:
                peaks[ti,:len(pk_freqs)] = pk_freqs

        # collect in to DataFrame
        col_names = [sig_comp+'pk'+str(i) for i in range(n_peaks)]
        #print len(peaks)
        for i in range(n_peaks):
            #feats = pd.DataFrame(peaks, columns=col_names)
            feats[col_names[i]] = peaks[:,i]
    #print len(feats), len(acts)
    #feats['act'] = acts
    
    return feats


def extract_spec_features(x, nFFT=256, n_peaks=3, delta=50):
    fs = 52.
    novr = nFFT/2
    nperseg = nFFT
    #pk1,pk2 = get_spec_peaks(Dat.mag, novr=0)

    feats = pd.DataFrame()
    # Calculate the spectrogram
    f,t,Sxx = spectrogram(x, fs=fs, nfft=nFFT, noverlap=novr, nperseg=nFFT)
    #print 'Sxx.shape', Sxx.shape
    n_nows = len(t)

    if 0:
        # get activity labels
        inds = np.arange(nFFT/2., nFFT*t.shape[0], nFFT)
        inds = [int(i) for i in inds]
        # TODO: should fix to not depend on processing actions.
        acts = [Dat.act[i] for i in inds]
        #acts.reset_index()

    # initialize for number of peaks to return
    peaks = np.zeros((len(t), n_peaks))
    # Find peaks for each time slice
    for ti in range(t.shape[0]):
        #print ti
        mx, mn = peakdet(Sxx[:,ti], delta=delta)
        mx, mn = mx.astype(int), mn.astype(int)
        # only concered with maxima for frequency domain features.
        pk_inds = [i[0] for i in mx]
        #print pk_inds
        pk_freqs = [f[i] for i in pk_inds]
        #print pk_freqs
        if len(pk_freqs) >= n_peaks:
            peaks[ti,:] = pk_freqs[:n_peaks]
        else:
            peaks[ti,:len(pk_freqs)] = pk_freqs

    return peaks


def time_feature_hist(X, nbins=40, lgnd=['diff_mx','diff_mn','diff_adj']):
    plt.subplot(131)
    plt.xlabel(lgnd[0])
    plt.hist([i[0] for i in X], bins=nbins)
    plt.subplot(132)
    plt.xlabel(lgnd[1])
    plt.hist([i[1] for i in X], bins=nbins)
    plt.subplot(133)
    plt.xlabel(lgnd[2])
    plt.hist([i[2] for i in X], bins=nbins)
    plt.show()

def time_feature_scatter(X):
    plt.subplot(221)
    plt.scatter([i[0] for i in X], [i[1] for i in X])
    plt.subplot(223)
    plt.scatter([i[0] for i in X], [i[2] for i in X])
    plt.subplot(224)
    plt.scatter([i[1] for i in X], [i[2] for i in X])
    plt.show()


def extract_windowed_time_features(dat, ts, win_size, delta, typ='amp', jrk=1):
    #print ts.shape
    X = dat.as_matrix()
    
    win_size_samp = int(win_size/(ts[1]-ts[0]))
    n_wins = X.shape[0]/win_size_samp

    if jrk:
        rslt = np.empty([0,12])
        lbl = ['AccPkMn', 'AccVlMn', 'AccAjMn', 'AccPkSd', 'AccVlSd', 'AccAjSd',
            'JrkPkMn', 'JrkVlMn', 'JrkAjMn', 'JrkPkSd', 'JrkVlSd', 'JrkAjSd']
    else:
        rslt = np.empty([0,6])
        lbl = ['AccPkMn', 'AccVlMn', 'AccAjMn', 'AccPkSd', 'AccVlSd', 'AccAjSd']
        
    
    lbls = []

    # dat has one signal column
    if len(X.shape) == 1:
        lbls = lbl
        x = X[:n_wins*win_size_samp]
        x = x.reshape((n_wins, win_size_samp))
        rslt = np.apply_along_axis(
            compute_time_stats, 1, x, ts=ts, delta=delta)
    # dat has multiple signal columns. Compute stats for each
    # and collect results.
    else:
        rslt = np.empty([n_wins,0])
        for i in range(X.shape[1]):
            #sig = dat.columns[i]
            sig = dat.columns[i]
            #print sig
            lbls.extend([sig+j for j in lbl])
        
            x = X[:n_wins*win_size_samp, i]
            x = x.reshape((n_wins, win_size_samp))
            
            r = np.apply_along_axis(
                compute_time_stats, 1, x, ts=ts, delta=delta, typ=typ, jrk=jrk)
            rslt = np.hstack((rslt, r))
            #print lbls
            #print '*', rslt.shape
            #print r.shape

    #print len(lbls)
    result = pd.DataFrame(rslt)#, columns=lbls)

    return result



def compute_time_stats(x, ts, delta=25, typ='amp', jrk=1):
    """Takes np.array, not pd.DataFrame
    """
    #jrk = 0
    #diffs_acc, diffs_jrk = extract_time_features(x,ts,delta)
    if typ == 'amp':
        diffs_acc = calculate_ts_amp_diffs(x, delta=delta)
        if jrk:
            diffs_jrk = calculate_ts_amp_diffs(np.diff(x), delta=delta*.75)
    else:
        diffs_acc = calculate_ts_diffs(x, ts, delta=delta)
        if jrk:
            diffs_jrk = calculate_ts_diffs(np.diff(x), ts[1:], delta=delta*.75)

    diffs_acc = np.array(diffs_acc)
    if jrk:
        diffs_jrk = np.array(diffs_jrk)
    

    means_acc = diffs_acc.mean(axis=0)
    stds_acc = diffs_acc.std(axis=0)
    if jrk:
        means_jrk = diffs_jrk.mean(axis=0)
        stds_jrk = diffs_jrk.std(axis=0)
    #print means_acc, stds_acc, means_jrk, stds_jrk
    #return np.mean(diffs_acc), np.std(diffs_acc), np.mean(diffs_jrk), np.std(diffs_jrk)
    
    if jrk:
        #print means_acc.shape, stds_acc.shape, means_jrk.shape, stds_jrk.shape
        rslt = np.concatenate((means_acc, stds_acc, means_jrk, stds_jrk))
    else:
        #print means_acc.shape, stds_acc.shape
        rslt = np.concatenate((means_acc, stds_acc))


    return rslt


def calculate_ts_diffs(x, ts, delta=25, viz=0):
    '''Old version that computed temporal/periodicity differences in the 
    signal.  This was a misinterpretation of the journal article.  But the 
    code is being kept for reference.
    '''
    #print "ts_diffs: x.shape:", x.shape
    #print "delta", delta
    mx, mn = peakdet(x, delta=delta)
    #print len(mx), len(mn) 
    mx, mn = mx.astype(int), mn.astype(int)
    if len(mx) > len(mn):
        mx = mx[:len(mn)]
    elif len(mn) > len(mx):
        mn = mn[:len(mx)]

    #print len(mx), len(mn)
    if viz:
        fig = plt.figure(figsize=(8,3))
        plt.plot(ts,x, 'k')
        for i in mx:
            plt.plot(ts[i[0]], i[1], 'bo')
        for i in mn:
            plt.plot(ts[i[0]], i[1], 'ro')
        plt.xlabel('Time (s)')
        plt.ylabel('Signal Amplitude')
        fig.tight_layout()
        plt.show()

    diff_mx = [ts[j[0]]-ts[i[0]] for i,j in zip(mx[:-1], mx[1:])]
    diff_mn = [ts[j[0]]-ts[i[0]] for i,j in zip(mn[:-1], mn[1:])]
    diff_adj = [0,0]
    if len(mx) > 1:
        if mx[0][0] < mn[0][0]:
            diff_adj = [ts[j[0]]-ts[i[0]] for i,j in zip(mx[1:], mn[1:])]
        else:
            diff_adj = [ts[j[0]]-ts[i[0]] for i,j in zip(mn[1:], mx[1:])]
    else:
        diff_mx = [0,0,0]
        diff_mn = [0,0,0]
        diff_adj = [0,0,0]

    #print len(diff_mx), len(diff_mn), len(diff_adj)
    rslt = [[a,b,c] for a,b,c in zip(diff_mx, diff_mn, diff_adj)]
    #print "rstl:",rslt
    return rslt

def calculate_ts_amp_diffs(x, delta=25, viz=0):
    """Takes a signal and returns:
     - difference between pairs of consecutive peaks
     - difference between the value of consecutive upper-side peaks
     - difference between the value of consecutive lower-side peaks
     """
    mx, mn = peakdet(x, delta=delta)

    mx, mn = mx.astype(int), mn.astype(int)
    if len(mx) > len(mn):
        mx = mx[:len(mn)]
    elif len(mn) > len(mx):
        mn = mn[:len(mx)]

    if viz:
        ts = np.arange(0, x.shape[0]/52., 1/52.)
        plt.plot(ts, x, 'k')
        for i in mx:
            plt.plot(ts[i[0]], i[1], 'bo')
        for i in mn:
            plt.plot(ts[i[0]], i[1], 'ro')
        plt.xlabel('Time (s)')
        plt.ylabel('Acceleration')
        plt.tight_layout()
        plt.show()


    diff_mx = [j[1]-i[1] for i,j in zip(mx[:-1], mx[1:])]
    diff_mn = [j[1]-i[1] for i,j in zip(mn[:-1], mn[1:])]
    
    if len(mx) > 1:
        if mx[0][0] < mn[0][0]:
            diff_adj = [j[1]-i[1] for i,j in zip(mx[1:], mn[1:])]
        else:
            diff_adj = [j[1]-i[1] for i,j in zip(mn[1:], mx[1:])]
    else:
        diff_mx = [0,0,0]
        diff_mn = [0,0,0]
        diff_adj = [0,0,0]

    rslt = [[a,b,c] for a,b,c in zip(diff_mx, diff_mn, diff_adj)]
    #print "rstl:",rslt
    return rslt



def peaks_for_all(data_files):
    '''calculates spectogram peaks for all files'''
    peak1, peak2, subjn = [], [], []
    for n, fn in enumerate(data_files):
        print os.path.basename(fn)
        #dat = pd.read_csv(fn, header=0, names=col_names)
        dat = load_file(fn, act=4)
        pk1,pk2 = get_spec_peaks(dat.mag, novr=0)
        #ts1 = np.arange(nFFT/2, nFFT*)
    
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


def plot_all_peaks(data_files):
    #nFFT = 128
    results = peaks_for_all(data_files)
    plot_peaks(results)


def adj_segs(segs, x):
    segs[0][2] += x
    for i in range(1, len(segs)-1):
        segs[i][1] += x
        segs[i][2] += x
    segs[-1][1] += x
    return segs




def pltsegs(ax, segs, fs=52., yd=[0,1], labs=1):
    for i,l in enumerate(segs):
        #xd = [l[1]/fs,l[1]/fs]
        xd = [l[2]/fs, l[2]/fs]
        yd = [0,fs/2]
        #print xd
        #plt.plot(xd, yd, linewidth=2)
        plt.axvline(xd[0], linewidth=2)

        y_lims = ax.get_ylim()
        #plt.text(xd[0],yd[1], str(i+1))
        if labs:
            plt.text(xd[0], y_lims[1], str(l[0]))


def spec_3a(dat, segs=None, nFFT=128, novr=0, fs=52.):   
    if segs is None:
        segs = get_activity_segments(dat)
        #print segs

    plt.figure(figsize=(20,6))
    ax = plt.subplot(311)
    plt.specgram(dat.xa, Fs=fs, NFFT=nFFT, noverlap=novr)
    pltsegs(ax, segs)
    
    ax = plt.subplot(312)
    plt.specgram(dat.ya, Fs=fs, NFFT=nFFT, noverlap=novr)
    pltsegs(ax, segs)
    
    ax = plt.subplot(313)
    plt.specgram(dat.za, Fs=fs, NFFT=nFFT, noverlap=novr)
    pltsegs(ax, segs)
    
    plt.show()


def acc_3a(dat, segs=None):
    #segs = get_activity_segments(dat)
    ns = dat.shape[0] 
    ts = np.linspace(0, ns/52., num=ns)

    plt.figure(figsize=(12,6))
    ax = plt.subplot(311)
    plt.plot(ts, dat.xa, 'k')
    plt.ylabel('Acc_X')
    if segs:
        pltsegs(ax, segs)

    ax = plt.subplot(312)
    plt.plot(ts, dat.ya, 'k')
    plt.ylabel('Acc_Y')
    if segs:
        pltsegs(ax, segs)
    
    ax = plt.subplot(313)
    plt.plot(ts, dat.za, 'k')
    plt.ylabel('Acc_Z')
    plt.xlabel('Time (s)')
    if segs:
        pltsegs(ax, segs)
    
    plt.tight_layout()
    plt.show()

def gather_data(data_files, act=None, sig_comps='mag', nfft=256, n_peaks=3):
    print "should avoid this"
    X = pd.DataFrame()
    for fn in data_files:
        #print os.path.basename(fn)
        dat = load_file(fn, act=act)
        d = get_spec_features(dat, sig_comps=sig_comps, nFFT=nfft, n_peaks=n_peaks)
        d['subj'] = [int(os.path.basename(fn)[:-4])] * len(d)

        X = X.append(d, ignore_index=True)

    return X


def compute_spec_features(data, act=None, sig_comps='mag', nfft=256, n_peaks=3):
    print "should avoid this"
    X = pd.DataFrame()
    for fn in data_files:
        #print os.path.basename(fn)
        subj_mask = data.subj.isin([i])
        dat = data[subj_mask]
        d = get_spec_features(dat, sig_comps=sig_comps, nFFT=nfft, n_peaks=n_peaks)
        d['subj'] = [int(os.path.basename(fn)[:-4])] * len(d)

        X = X.append(d, ignore_index=True)

    return X

def split_data(Dat, subjects=None, actions=[3,4], test_ratio=0.3, X_coi=[], y_coi='', 
    random_state=3):
    
    if subjects is None:
        subjects = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
    if X_coi == [] or y_coi == '':
        print "Specify columns of interest"
        return 0

    #X = Dat[X_coi]
    #y = Dat[y_coi]
    #X_train, X_test, y_train, y_test = train_test_split(
    #        X, y, test_size=test_ratio, random_state=random_state)
    
    X_train, y_train = pd.DataFrame(), pd.DataFrame([], columns=[y_coi])
    X_test, y_test = pd.DataFrame(), pd.DataFrame([], columns=[y_coi])

    for si in subjects:
        for ai in actions:
            d = Dat[(Dat.subj==si) & (Dat.act==ai)]
            n_rows = len(d)
            #print n_rows,
            n_test = int(n_rows*test_ratio)
            n_train = n_rows - n_test

            X_train = X_train.append(d[X_coi][:n_train])
            X_test = X_test.append(d[X_coi][n_train:n_rows])
            y_train = y_train.append(d[y_coi][:n_train])
            y_test = y_test.append(d[y_coi][n_train:n_rows])
            #print len(X_train), len(X_test)

    #return X, y
    return X_train, y_train.astype(int), X_test, y_test.astype(int)

