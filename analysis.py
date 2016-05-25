import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.learning_curve import learning_curve
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.mixture import GMM
from sklearn.metrics import confusion_matrix

from sklearn.metrics import f1_score
from scipy.signal import resample
from scipy.signal import get_window
from scipy.signal import spectrogram #, find_peaks_cwt
from scipy.signal import butter, lfilter
from peakdetect import peakdet
import time

from sklearn.cross_validation import train_test_split
from sklearn import svm
from sklearn import tree

from time_series_segmentation import peak_detection


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


def filter_signal(x, ts=None, cutoff=5., fs=52., order=6, viz=0):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    xf = lfilter(b, a, x)
    
    if viz and ts is not None:
        #t = np.linspace(0, len(x)/, len(x), endpoint=False)
        plt.plot(ts, x, 'b-', label='data')
        plt.plot(ts, xf, 'g-', label='filtered')
        plt.xlabel('Time (s)')
        plt.legend()
        plt.show()


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


def plt_walking_psd(data, sig, nFFT=256, pk_dist=.3, show_peaks=False):
    plt.figure()
    for si in list(set(data.subj)):
        di = data[data.subj == si]
        di = di[di.act == 4]
        x = di[sig][:]
        a,b,l = plt.psd(x, nFFT, 52., return_line=True)
        # grab data from the plot
        pwr, fq = l[0].get_ydata(), l[0].get_xdata()
        p = peak_detection(pwr, 6, 0, .02, int(pk_dist*52))
        #print pwr, fq   
        if show_peaks:
            for pi in p:
                #print pi
                plt.plot(fq[pi[0]], pi[1], 'ko')

    plt.show()
    return pwr, fq

def plt_psd_w_peaks(f,x, delta=10):
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


def get_spec_peaks(dat, n_peaks=8, nFFT=128, fs=52, novr=0, nperseg=128):
    '''returns spetrogram peaks for one time series'''


    #sp = plt.specgram(dat, NFFT=nFFT, noverlap=novr)
    f,t,Sxx = spectrogram(dat, fs=fs, nfft=nFFT, noverlap=novr, nperseg=nperseg)
    #plt.show()
    pk1 = []
    pk2 = []
    pks = []
    for ti in range(t.shape[0]):
        #p_ind = find_peaks(Sxx[:][t])
        p_ind = find_peaks(Sxx[ti][:])
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
    #pks = np.array(pks)
    
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
        # TODO: should fix to not depend on processing actions. 
        # just remove?
        #acts = [Dat.act[i] for i in inds]
        #acts.reset_index()

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
    novr = 0
    nperseg = nFFT
    #pk1,pk2 = get_spec_peaks(Dat.mag, novr=0)

    feats = pd.DataFrame()
    # Calculate the spectrogram
    f,t,Sxx = spectrogram(x, fs=fs, nfft=nFFT, noverlap=novr, nperseg=nperseg)
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


def build_time_domain_data(data_files):
    print 'stand alone function'
    act_n = 4
    X, y = np.empty([0,12]), np.empty([0,2])
    for fn in data_files:
        subj_n = int(os.path.basename(fn)[:-4])
        print subj_n,
        dat = load_file(fn, act=act_n)
        rslt = extract_windowed_time_features(
            dat.ya.as_matrix(), dat.ts.as_matrix(), 5, 25)
        X = np.concatenate((X, rslt), 0)
        #print rslt.shape
        #act_col = np.array([[act_n]] * rslt.shape[0])
        #subj_col = np.array([[subj_n]] * rslt.shape[0])
        y_cols = np.array([[act_n, subj_n]] * rslt.shape[0])
        y = np.concatenate((y, y_cols), 0)
        #print act_col.shape
        #print subj_col.shape
        #rslt = np.concatenate((rslt, act_col, subj_col), 1)
        #print rslt.shape
        #print result.shape
        #result = np.concatenate((result, rslt), 0)
    print ''

    return X, y

def compute_time_domain_data(data, sig_col='ya', window_size=5, delta=25):
    """not used?"""
    act_n = 4
    # there are 12 X features to compute and 2 labels
    X, y = np.empty([0,12]), np.empty([0,2])

    for s in subj_numbers:
        #subj_n = int(os.path.basename(fn)[:-4])
        print s,
        subj_mask = data.subj.isin([s])
        dat = data[subj_mask]
        #dat = load_file(fn, act=act_n)
        rslt = extract_windowed_time_features(
            dat[sig_col].as_matrix(), dat.ts.as_matrix(), window_size, delta)
        X = np.concatenate((X, rslt), 0)
        
        y_cols = np.array([[act_n, subj_n]] * rslt.shape[0])
        y = np.concatenate((y, y_cols), 0)
    print ''

    return X, y

def extract_windowed_time_features(dat, ts, win_size, delta):
    #print ts.shape
    X = dat.as_matrix()

    win_size_samp = int(win_size/(ts[1]-ts[0]))
    n_wins = X.shape[0]/win_size_samp

    rslt = np.empty([0,12])
    lbl = ['mnAcc', 'sdAcc', 'mnJrk', 'sdJrk']
    lbls = []

    if len(X.shape) == 1:
        lbls = lbl
        x = X[:n_wins*win_size_samp]
        x = x.reshape((n_wins, win_size_samp))
        rslt = np.apply_along_axis(
            extract_time_stats, 1, x, ts=ts, delta=delta)
    else:
        rslt = np.empty([n_wins,0])
        for i in range(X.shape[1]):
            #sig = dat.columns[i]
            sig = dat.columns[i]
            lbls.extend([sig+j for j in lbl])
        
            x = X[:n_wins*win_size_samp, i]
            x = x.reshape((n_wins, win_size_samp))
            #print "win shape:", x.shape
            #print x[0]
            #print x[1]
            r = np.apply_along_axis(
                extract_time_stats, 1, x, ts=ts, delta=delta)
            rslt = np.hstack((rslt, r))
            #print r.shape

    #print lbls
    result = pd.DataFrame(rslt)#, columns=lbls)

    return result



def extract_time_stats(x, ts, delta=25):
    """Takes np.array, not pd.DataFrame
    """
    #diffs_acc, diffs_jrk = extract_time_features(x,ts,delta)
    diffs_acc = calculate_ts_diffs(x, ts, delta=delta)
    diffs_jrk = calculate_ts_diffs(np.diff(x), ts[1:], delta=delta*.75)

    diffs_acc, diffs_jrk = np.array(diffs_acc), np.array(diffs_jrk)
    

    means_acc = diffs_acc.mean(axis=0)
    stds_acc = diffs_acc.std(axis=0)
    means_jrk = diffs_jrk.mean(axis=0)
    stds_jrk = diffs_jrk.std(axis=0)
    #print means_acc, stds_acc, means_jrk, stds_jrk
    #return np.mean(diffs_acc), np.std(diffs_acc), np.mean(diffs_jrk), np.std(diffs_jrk)
    rslt = np.concatenate((means_acc, stds_acc, means_jrk, stds_jrk))

    return rslt


def extract_time_features(x, ts, delta=25):
    """Takes np.array, not pd.DataFrame
    """
    # DONE: move this functionality to extract_time_stats
    #x = x.as_matrix()
    #ts = ts.as_matrix()
    diffs_acc = calculate_ts_diffs(x, ts, delta=delta)
    #jrk_sig = x[1:].as_matrix() - x[:-1].as_matrix()
    #jrk_sig = x[1:] - x[:-1]
    jrk_sig = np.diff(x)
    #print "jrk shape:", jrk_sig.shape
    diffs_jrk = calculate_ts_diffs(jrk_sig, ts[1:], delta=int(delta*.75))
    #print len(diffs_acc), len(diffs_jrk)

    return diffs_acc, diffs_jrk
    #return np.mean(diffs_acc)


def calculate_ts_diffs(x, ts, delta=25):
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
    if 0:
        plt.plot(Dat[sig_comp], 'k')
        for i in mx:
            plt.plot(i[0], i[1], 'bo')
        for i in mn:
            plt.plot(i[0], i[1], 'ro')
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

def calculate_ts_amp_diffs(x, delta=25):
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

    if 0:
        plt.plot(x, 'k')
        for i in mx:
            plt.plot(i[0], i[1], 'bo')
        for i in mn:
            plt.plot(i[0], i[1], 'ro')
        plt.show()


    diff_mx = [j[1]-i[1] for i,j in zip(mx[:-1], mx[1:])]
    diff_mn = [j[1]-i[1] for i,j in zip(mn[:-1], mn[1:])]
    diff_adj = [0,0]
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



def find_peaks_x(dat):
    pks = np.arange(1,50)
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
        print segs

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


def plot_all_peaks(data_files):
    #nFFT = 128
    results = peaks_for_all(data_files)
    plot_peaks(results)

def analysis_first(Dat):
    # First analysis as sanity check
    #Dat = gather_data(data_files)
    Dat = Dat.query('(act == 4) | (act == 3)')
    X_train, y_train, X_test, y_test = split_data(
        Dat, X_coi=['pk0', 'pk1', 'pk2'], y_coi=['act'])
    clf = svm.SVC()
    clf.fit(X_train, np.ravel(y_train))
    #print clf
    print clf.score(X_test, y_test)

    return clf

def analysis_by_nfft(data_files, clf):
    ffts = [128, 192, 256, 384, 512]
    ffts = [128, 256, 512]
    xcoi = ['pk0', 'pk1', 'pk2']
    #clf = svm.SVC()

    results = np.zeros((len(ffts),4))
    for i, sig in enumerate(['xa', 'ya', 'za', 'mag']):
        print "Signal", sig
        for j, fft_ in enumerate(ffts):
            print "FFT size:", fft_
            Dat = gather_data(data_files, sig_comp=sig, nfft=fft_)
            X_train, y_train, X_test, y_test = split_data(Dat, actions=[3,4], 
                test_ratio=0.1, X_coi=xcoi, y_coi=['act'])
            y_train = np.ravel(y_train)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            f1 = f1_score(y_pred, y_test, pos_label=4)
            print "F1 score:", f1
            results[j][i] = f1
        print

    return results


def analysis_walking_identification(clf, Dat, subjs=[1,2]):
    #xcoi = ['pk0', 'pk1', 'pk2']
    xcoi = [i for i in Dat.columns if i not in ['act', 'subj']]

    for lo in range(1,max(subjs)+1):
        print 'leave out', lo
        # get train/test data
        X, y, _, _ = split_data(Dat, subjects=subjs, actions=[4], 
            test_ratio=0.0, X_coi=xcoi, y_coi=['subj'])
        #print len(y_train), len(y_test)
        y = np.ravel(y)
        #y_test = np.ravel(y_test)
        #y = y[y!=1]

        # revise y_labels for Leave One Out Analysis?
        y[y != lo] = 0
        #y_test[y_test != lo] = 0
        
        clf.fit(X, y)
        scores = cross_val_score(clf, X, y, cv=3)
        print scores.mean()

    #plt.scatter(X[])
    return X, y


def learning_curve_analysis(Dat, acts=[3,4]):
    #acts = [2,3,4,5]
    xcoi = [i for i in Dat.columns if i not in ['act', 'subj']]

    X, y,_, _= split_data(Dat, test_ratio=0., actions=acts,
        X_coi=xcoi, 
        y_coi=['act'])#, 'subj'])
    y = np.ravel(y)
    plot_learning_curve(svm.SVC(), "SVC Learning Curve", X, y, 
        train_sizes=np.linspace(.2, 1., 5))
    plt.show()


def analysis_svm(X_train, y_train, X_test, y_test):
    clf = svm.SVC()
    clf.fit(X_train, y_train)


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and traning learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : integer, cross-validation generator, optional
        If an integer is passed, it is the number of folds (defaults to 3).
        Specific cross-validation objects can be passed, see
        sklearn.cross_validation module for the list of possible objects

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt


def split_Xy(X, y, subjects=None, actions=None, test_ratio=0.3, random_state=3):
    
    y_items = set(y.tolist())

    #if subjects is None:
    #    subjects = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
    #if X_coi == [] or y_coi == '':
    #    print "Specify columns of interest"
    #    return 0

    #X = Dat[X_coi]
    #y = Dat[y_coi]
    #X_train, X_test, y_train, y_test = train_test_split(
    #        X, y, test_size=test_ratio, random_state=random_state)
    
    #X_train, y_train = pd.DataFrame(), pd.DataFrame()
    #X_test, y_test = pd.DataFrame(), pd.DataFrame()
    X_train, y_train = np.empty([0,X.shape[1]]), np.empty([1,0])
    X_test, y_test = np.empty([0,X.shape[1]]), np.empty([1,0])

    for yi in y_items:
        #d = Dat[(y==si)]
        i_mask = y==yi
        n_rows = len(y[i_mask])
        n_test = int(n_rows*test_ratio)
        n_train = n_rows - n_test
        #print n_rows,
        Xi = X[i_mask]
        yi = y[i_mask]
        
        X_train = np.vstack((X_train, Xi[:][:n_train]))
        X_test = np.vstack((X_test, Xi[:][n_train:n_rows]))
        y_train = np.append(y_train, yi[:n_train])
        y_test = np.append(y_test, yi[n_train:n_rows])
        #print len(X_train), len(X_test)

    #return X, y
    return X_train, y_train.astype(int), X_test, y_test.astype(int)

def plot_confusion_matrix(cm, target_names, title='Confusion matrix', cmap=plt.cm.Blues):
    #target_names = ['Stairs', 'Standing', 'Walking', 'Talking']
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(target_names))
    plt.xticks(tick_marks, target_names, rotation=45)
    plt.yticks(tick_marks, target_names)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def analysis_classify_activity(clf, data, sig_comp='ya'):
    subj_n = range(1,16)
    activities_list = ['Working at Computer', 'Stairs', 'Standing', 'Walking',
        'Stairs', 'Walking and Talking', 'Talking while Standing']
    act_n = [3,4]
    act_n = [2,3,4,7]
    activities = [activities_list[i-1] for i in act_n]

    #print data.head()
    if 1:
        # time domain features
        if type(sig_comp) == 'str':
            X, y = np.empty([0,12]), np.empty([1,0])
        else:#if type(sig_comp) == 'list':
            X, y = np.empty([0,12*len(sig_comp)]), np.empty([1,0])

        print "Extracting time features..."
        t = time.time()
        for i in act_n:
            d = data[data.act.isin([i])]
            #print d.head()

            f = extract_windowed_time_features(
                d[sig_comp].as_matrix(), d.ts.as_matrix(), 2, 50)

            #f = pd.DataFrame(f, columns=)
            #print i, f.shape[0]
            act_col = [i] * f.shape[0]
            #print subj_col
            #f.subj = subj_col
            y = np.append(y, act_col)
            X = np.vstack((X, f))
            #feats_time.append(f)
        y = y.astype(int)
        print "Time:", time.time() - t
        if type(sig_comp) == 'list':
            print 'pca reduction...'
            pca = PCA(n_components=10)
            #pca.fit(X)
            X = pca.fit_transform(X)

    else:
        # frequency domain features
        print "Extracting frequency features..."
        t = time.time()
        X, y = np.empty([0,3]), np.empty([1,0])
        for i in act_n:
            d = data[data.subj.isin([i])]

            f = extract_spec_features(d[sig_comp].as_matrix(), 
                nFFT=256, n_peaks=3, delta=3)
            #f['subj'] = [i] * f.shape[0]
            #feats_freq.append(f)
            act_col = [i] * f.shape[0]
            y = np.append(y, act_col)
            X = np.vstack((X, f))
        y = y.astype(int)
        print "Time:", time.time() - t


    #scores = cross_val_score(clf, X_time, y_time)
    #print scores
    X_train, y_train, X_test, y_test = split_Xy(X, y, test_ratio=.3)
    #print y.shape
    #sss = train_test_split(X, y, 
    #    train_size=.3, stratify=y)
    
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    if 1:
        cm = confusion_matrix(y_test, y_pred)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:,np.newaxis]
        print('Normalized confusion matrix')
        print(cm_normalized)
        plt.figure()
        plot_confusion_matrix(cm_normalized, activities, title='Normalized confusion matrix')

        plt.show()
    return clf


def analysis_compare_time_freq(clf, data):
    # to be deleted
    subj_n = range(1,16)

    if 0:
        X_time, y_time = make_time_features(data, win_size=2, delta=40)
        pca_time = PCA()
        X_time_pca = pca_time.fit_transform(X_time)
        plt.figure()
        plt.title('PCA time features')
        plt.plot(np.cumsum(pca_time.explained_variance_ratio_))
        lo_time = analysis_classify_walkers(clf, X_time, y_time)

    X_freq, y_freq = make_freq_features(data, nFFT=256, n_peaks=5, delta=50)

    

    
    
    pca_freq = PCA()
    X_freq_pca = pca_freq.fit_transform(X_freq)
    plt.figure()
    plt.title('PCA freq features')
    plt.plot(np.cumsum(pca_freq.explained_variance_ratio_))
    
    #return (X_time,y_time), (X_freq, y_freq)
    
    lo_freq = analysis_classify_walkers(clf, X_freq, y_freq)

    if 0:
        plt.figure()
        plt.plot(lo_time, label='Time Features')
        plt.plot(lo_freq, label='Frequency Features')
        plt.legend()
        plt.show()

    return (X_time, y_time), (X_freq, y_freq)


def compare_classifiers(X, y):

    clfs = [#('AdaBoost', AdaBoostClassifier(n_estimators=50)),
            #('KNN', NearestNeighbors(n_neighbors=3)),
            ('KMeans', KMeans(n_clusters=2)),
            ('GMM', GMM(n_components=2))
        ]

    for name, clf in clfs:
        print name,':',
        rslt = analysis_classify_walkers_louo(clf, X, y)
        print rslt


def analysis_classify_walkers(clf, X, y):
    scores = cross_val_score(clf, X, y)
    print scores

def analysis_classify_walkers_louo(clf, X, y):
    #xcoi = ['pk0', 'pk1', 'pk2']
    #xcoi = [i for i in Dat.columns if i not in ['act', 'subj']]

    lo_scores = []
    for lo in range(1,max(y)+1):
        #print 'leave out', lo
        yi = np.copy(y)
        
        # revise y_labels for Leave One Out Analysis?
        yi[yi != lo] = 0
        #y_test[y_test != lo] = 0
        
        clf.fit(X, yi)
        scores = cross_val_score(clf, X, yi, cv=3)
        #print scores.mean()
        lo_scores.append(scores.mean())

    #plt.scatter(X[])
    scores = np.array(lo_scores)
    return scores.mean(), scores.std()


def plot_as_pca(X):
    pca = PCA(n_components=2)
    Xt = pca.fit_transform(X[0])

    for i in range(1,6):
        print i,
        Xi = Xt[np.ravel(X[1]==i)]
        Xc = np.ravel(X[1][np.ravel(X[1]==i)])
        print Xc
        print Xi.shape
        plt.scatter(Xi[:,0], Xi[:,1], c=Xc)#[i]*Xi[0].size)
    plt.show()

def plot_windowed_time_features(data_file, n, sig='ya', win_size=2):
    dat = load_data(data_file[n:n+1], act=4)
    r = extract_windowed_time_features(
        dat.ya.as_matrix(), dat.ts.as_matrix(), win_size, 40)
    plt.plot(r)
    plt.show()

def make_freq_features(data, nFFT=256, n_peaks=6, delta=40,
    yrng=range(1,16), ycol='subj'):

    #subj_n = range(1,16)#[1]
    sig_comps = ['xa', 'ya', 'za']
    n_sig = len(sig_comps)

    # frequency domain features
    print "Extracting frequency features..."
    t = time.time()
    #X_freq, y_freq = np.empty([0,n_peaks]), np.empty([1,0])
    #X_freq, y_freq = np.empty([0,n_peaks*n_sig]), np.empty([1,0])
    X = pd.DataFrame()
    y = np.array([])
    for i in yrng: #subj_n:
        dat = data[data[ycol].isin([i])]
        
        f = get_spec_features(dat, sig_comps,
                nFFT=nFFT, n_peaks=n_peaks, delta=delta)
        
        y_col = pd.DataFrame({'subj': [i] * f.shape[0]})
        y = np.append(y, y_col)#, ignore_index=True)
        X = X.append(f, ignore_index=True)

    y = y.astype(int)
    print "Time:", time.time() - t

    return X, y

def make_time_features(data, win_size=2, delta=40, yrng=range(1,16), ycol='subj'): 

    #subj_n = range(1,16)#[1]
    sig_comps = ['xa', 'ya', 'za']
    n_sig = len(sig_comps)

    X = pd.DataFrame()
    y = np.array([])
    
    print "Extracting time features..."
    t = time.time()
    for i in yrng: #subj_n:
        d = data[data[ycol].isin([i])]

        f = extract_windowed_time_features(
            d[sig_comps], d.ts.as_matrix(), win_size, delta)
        y_col = pd.DataFrame({ycol: [i] * f.shape[0]})
        
        y = np.append(y, y_col) #, ignore_index=True)
        X = X.append(f, ignore_index=True)

    y = y.astype(int)
    print "Time:", time.time() - t

    return X, y

'''
Analyses
'''


def analysis_time_ada(data):
    walking_data = data[data.act==4]
    clf = AdaBoostClassifier(n_estimators=60)
    X, y = make_time_features(walking_data)
    
    print 'Walker Classification.'
    analysis_classify_walkers(clf, X, y)
    
    print 'Leave One User Out.'
    mn, sd = analysis_classify_walkers_louo(clf, X, y)
    print 'mean',mn, 'std', sd
    

def analysis_freq_ada(data):
    walking_data = data[data.act==4]
    clf = AdaBoostClassifier(n_estimators=60)
    X, y = make_freq_features(walking_data)

    print 'Walker Classification.'
    analysis_classify_walkers(clf, X, y)
    
    print 'Leave One User Out.'
    mn, sd = analysis_classify_walkers_louo(clf, X, y)
    print 'mean',mn, 'std', sd


def analysis_time_tree(data):
    walking_data = data[data.act==4]
    clf = tree.DecisionTreeClassifier()
    if 0:
        X, y = make_time_features(walking_data)
        
        print 'Walker Classification.'
        analysis_classify_walkers(clf, X, y)
        
        print 'Leave One User Out.'
        mn, sd = analysis_classify_walkers_louo(clf, X, y)
        print 'mean',mn, 'std', sd
    X, y = make
    
    
def analysis_freq_tree(data):
    walking_data = data[data.act==4]
    clf = tree.DecisionTreeClassifier()
    X, y = make_freq_features(walking_data)

    print 'Walker Classification.'
    analysis_classify_walkers(clf, X, y)
    
    print 'Leave One User Out.'
    mn, sd = analysis_classify_walkers_louo(clf, X, y)
    print 'mean',mn, 'std', sd

def analysis_time_svc(data):
    walking_data = data[data.act==4]
    clf = svm.SVC()
    X, y = make_time_features(walking_data)
    
    print 'Walker Classification.'
    analysis_classify_walkers(clf, X, y)
    
    print 'Leave One User Out.'
    mn, sd = analysis_classify_walkers_louo(clf, X, y)
    print 'mean',mn, 'std', sd
    
    
def analysis_freq_svc(data):
    walking_data = data[data.act==4]
    clf = svm.SVC()
    X, y = make_freq_features(walking_data)

    print 'Walker Classification.'
    analysis_classify_walkers(clf, X, y)
    
    print 'Leave One User Out.'
    mn, sd = analysis_classify_walkers_louo(clf, X, y)
    print 'mean',mn, 'std', sd



