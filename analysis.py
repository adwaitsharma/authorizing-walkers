import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.learning_curve import learning_curve
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
    ts = np.arange(0, dat.shape[0]/52., 1/52.)
    dat['ts'] = ts

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

def plt_psd_w_peaks(f,x, delta=10):
    #p_ind = find_peaks(x)
    mx, mn = peakdet(x, delta=delta)
    #print mx
    #print mn
    plt.plot(x)
    for p in mx:
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

def get_spec_features(Dat, sig_comps='mag', nFFT=256, n_peaks=3):
    fs = 52.
    novr = 0
    nperseg = nFFT
    delta = 50
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

        # collect in to DataFrame
        col_names = [sig_comp+'pk'+str(i) for i in range(n_peaks)]
        #print len(peaks)
        for i in range(n_peaks):
            #feats = pd.DataFrame(peaks, columns=col_names)
            feats[col_names[i]] = peaks[:,i]
    #print len(feats), len(acts)
    feats['act'] = acts
    
    return feats


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


def extract_time_features(x, ts, delta=25):
    #x = x.as_matrix()
    ts = ts.as_matrix()
    diffs_acc = calculate_ts_diffs(x, ts, delta=delta)
    jrk_sig = x[1:].as_matrix() - x[:-1].as_matrix()
    print jrk_sig.shape
    diffs_jrk = calculate_ts_diffs(jrk_sig, ts[1:], delta=int(delta*.75))
    print len(diffs_acc), len(diffs_jrk)

    return diffs_acc, diffs_jrk


def calculate_ts_diffs(x, ts, delta=25):
    mx, mn = peakdet(x, delta=delta)
    mx, mn = mx.astype(int), mn.astype(int)
    if len(mx) > len(mn):
        mx = mx[:len(mn)]
    elif len(mn) > len(mx):
        mn = mn[:len(mx)]

    print len(mx), len(mn)
    if 0:
        plt.plot(Dat[sig_comp], 'k')
        for i in mx:
            plt.plot(i[0], i[1], 'bo')
        for i in mn:
            plt.plot(i[0], i[1], 'ro')
        plt.show()

    diff_mx = [ts[j[0]]-ts[i[0]] for i,j in zip(mx[:-1], mx[1:])]
    diff_mn = [ts[j[0]]-ts[i[0]] for i,j in zip(mn[:-1], mn[1:])]

    if mx[0][0] < mn[0][0]:
        diff_adj = [ts[j[0]]-ts[i[0]] for i,j in zip(mx[1:], mn[1:])]
    else:
        diff_adj = [ts[j[0]]-ts[i[0]] for i,j in zip(mn[1:], mx[1:])]

    print len(diff_mx), len(diff_mn), len(diff_adj)

    return [[a,b,c] for a,b,c in zip(diff_mx, diff_mn, diff_adj)]





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



def find_peaks(dat):
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
    X = pd.DataFrame()
    for fn in data_files:
        #print os.path.basename(fn)
        dat = load_file(fn, act=act)
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
            X_test = X_test.append(d[X_coi][-n_test:])
            y_train = y_train.append(d[y_coi][:n_train])
            y_test = y_test.append(d[y_coi][-n_test:])
            #print len(X_test)

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


def analysis_walking_identification(Dat, subjs=[1,2]):
    xcoi = ['pk0', 'pk1', 'pk2']
    xcoi = [i for i in Dat.columns if i not in ['act', 'subj']]
    # get train/test data
    X_train, y_train, X_test, y_test = split_data(Dat, subjects=subjs, actions=[4], 
        test_ratio=0., X_coi=xcoi, y_coi=['subj'])
    print len(y_train), len(y_test)
    y_train = np.ravel(y_train)
    y_test = np.ravel(y_test)
    #y_train = y_train[y_train!=1]

    # revise y_labels for Leave One Out Analysis?
    y_train[y_train != 1] = 0
    y_test[y_test != 1] = 0
    #print y_train
    #print y_test


    clf = svm.SVC()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    f1 = f1_score(y_pred, y_test, pos_label=1)

    print "F1 score:", f1

    return clf


    
    

def learning_curve_analysis(Dat, acts=[3,4]):
    #acts = [2,3,4,5]
    X, y,_, _= split_data(Dat, test_ratio=0., actions=acts,
        X_coi=['pk0', 'pk1', 'pk2'], 
        y_coi=['act'])#, 'subj'])
    y = np.ravel(y)
    plot_learning_curve(svm.SVC(), "SVC Learning Curve", X, y, 
        train_sizes=np.linspace(.1, .5, 5))
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