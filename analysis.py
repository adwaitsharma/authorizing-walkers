from algorithms import *
import time

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.tools.plotting import scatter_matrix

from scipy.stats import ttest_ind

from sklearn.cross_validation import train_test_split
from sklearn import svm
from sklearn import tree
from sklearn.linear_model import LogisticRegression

from sklearn.mixture import GMM
from sklearn.cluster import KMeans

#
# Get started!
#

# column names
col_names = ['ts','xa','ya','za','act']
# xyz motion columns
xyz = ['xa','ya','za']




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


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):#[50,100,150,200,300,400]) 

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
    print estimator
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
    act_n = [1,3,4]

    activities = [activities_list[i-1] for i in act_n]

    #print data.head()
    if 1:
        # time domain features
        if type(sig_comp) == 'str':
            X, y = np.empty([0,12]), np.empty([1,0])
        else:#if type(sig_comp) == 'list':
            X, y = np.empty([0,12*len(sig_comp)]), np.empty([1,0])

            '''print "Extracting time features..."
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
            '''
            X, y = make_time_features(data[data.subj==1], ycol='act', yrng=act_n)
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

    X_freq, y_freq = make_freq_features(data, nFFT=256, n_peaks=5, delta=50)

    

    
    
    pca_freq = PCA()
    X_freq_pca = pca_freq.fit_transform(X_freq)
    plt.figure()
    plt.title('PCA freq features')
    plt.plot(np.cumsum(pca_freq.explained_variance_ratio_))
    
    #return (X_time,y_time), (X_freq, y_freq)
    

    if 0:
        plt.figure()
        plt.plot(lo_time, label='Time Features')
        plt.plot(lo_freq, label='Frequency Features')
        plt.legend()
        plt.show()

    #return (X_time, y_time), (X_freq, y_freq)
    return (X_freq, y_freq)




def analysis_classify_walkers_louo(clf, X, y, parms={}):
    # list of scores for each iteration of user verification
    print 'LOUO'
    lo_scores = []

    for lo in range(1,max(y)+1):
        # revise y_labels for Leave One Out Analysis
        yi = np.copy(y)
        yi[yi != lo] = 0
        yi[yi == lo] = 1
        #print yi.mean()
        # make train/test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, yi, train_size=.7, random_state=3)
        clf.set_params(**parms)
        #print y_train.mean(), y_test.mean()

        # fit classifier and score classifier
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        lo_scores.append(score)
    print clf
    print 'train:', X_train.shape
    print 'test:', X_test.shape
    
    scores = np.array(lo_scores)
    return scores #.mean(), scores.std()


def plot_as_pca(X,y):
    pca = PCA(n_components=5)
    Xt = pca.fit_transform(X)
    import six 
    from matplotlib import colors
    colors_ = list(six.iteritems(colors.cnames))

    if 0:#for i in range(1,3):
        #print i,
        Xi = Xt[np.ravel(y==i)]
        Xc = np.ravel(y[np.ravel(y==i)])
        #print Xc
        #print Xi.shape
        #plt.scatter(Xi[:,0], Xi[:,1], c=Xc)#[i]*Xi[0].size)
        plt.scatter(Xi[:,4], Xi[:,2], c=colors_[i])#[i]*Xi[0].size)

        plt.show()


    Xtpd = pd.DataFrame(Xt, columns=['PCA '+str(i) for i in range(5)])
    #Xtpd['y'] = y
    scatter_matrix(Xtpd, alpha=0.2, figsize=(6, 6), diagonal='kde')

    return Xtpd

def plot_windowed_time_features(data_file, n, sig='ya', win_size=2):
    dat = load_data(data_file[n:n+1], act=4)
    r = extract_windowed_time_features(
        dat.ya.as_matrix(), dat.ts.as_matrix(), win_size, 40)
    plt.plot(r)
    plt.show()

def make_freq_features(data, nFFT=256, n_peaks=6, delta=4,
    yrng=range(1,16), ycol='subj'):

    #subj_n = range(1,16)#[1]
    sig_comps = ['xa', 'ya', 'za']
    #n_sig = len(sig_comps)

    # frequency domain features
    print "Extracting frequency features..."
    t = time.time()
    #X_freq, y_freq = np.empty([0,n_peaks]), np.empty([1,0])
    #X_freq, y_freq = np.empty([0,n_peaks*n_sig]), np.empty([1,0])
    X = pd.DataFrame()
    y = np.array([])
    for i in yrng: 
        dat = data[data[ycol].isin([i])]
        
        f = get_spec_features(dat, sig_comps,
                nFFT=nFFT, n_peaks=n_peaks, delta=delta)
        
        y_col = pd.DataFrame({'subj': [i] * f.shape[0]})
        y = np.append(y, y_col)#, ignore_index=True)
        X = X.append(f, ignore_index=True)

    y = y.astype(int)
    print "Time:", time.time() - t
    print 'Feature Matrix:', X.shape[0], 'rows,', X.shape[1], 'columns.'

    return X, y


def make_time_features(data, win_size=5, delta=40, yrng=range(1,16), ycol='subj', typ='amp', jrk=1): 

    #subj_n = range(1,16)#[1]
    sig_comps = ['xa', 'ya', 'za']
    #n_sig = len(sig_comps)

    X = pd.DataFrame()
    y = np.array([])
    
    print "Extracting time features..."
    t = time.time()
    for i in yrng: #subj_n:
        d = data[data[ycol].isin([i])]

        f = extract_windowed_time_features(
            d[sig_comps], d.ts.as_matrix(), win_size, delta, typ=typ, jrk=jrk)
        y_col = pd.DataFrame({ycol: [i] * f.shape[0]})
        
        y = np.append(y, y_col) #, ignore_index=True)
        X = X.append(f, ignore_index=True)

    y = y.astype(int)
    print "Time:", time.time() - t

    print 'Feature Matrix:', X.shape[0], 'rows,', X.shape[1], 'columns.'

    return X, y


'''*****************************************************************************
Analyses
*****************************************************************************'''

def analysis_clustering(data, n_clust=2):
    #n_clust = 2

    # make features
    X, y = make_freq_features(data, delta=40)
    #X, y = make_time_features(data, delta=40)

    # pca reduction
    pca = PCA(n_components=18)
    Xt = pca.fit_transform(X)
    if 0:
        plt.figure()
        plt.title('PCA freq features')
        plt.plot(range(1,19), np.cumsum(pca.explained_variance_ratio_), 'o-')
        plt.xlim(1,18)
        plt.show()

    pca = PCA(n_components=18)
    Xt = pca.fit_transform(X)
    
    max_pct = []
    clusts = range(3,16)
    for n_clust in clusts:
        # cluster into groups
        #clf = GMM(n_components=n_clust, covariance_type='full')
        clf = KMeans(n_clusters=n_clust)
        clf.fit(Xt)
        y_pred = clf.predict(Xt)

        # look at groups
        counts = []
        for c in range(n_clust):
            inds = np.where(y_pred==c)
            res = np.histogram(y[inds], bins=range(1,17))
            #print res[0]
            counts.append(res[0])

        #for i in range(n_clust):
        counts_tot = np.histogram(y, bins=range(1,17))[0]
        counts_tot = [float(i) for i in counts_tot]
        
        #pcts = [i/counts_tot for i in counts]
        pcts = [i/counts_tot*(a+2) for a,i in enumerate(counts)]
        #print pcts
        max_pct.append(np.max(pcts, 0))

    pct_dat = np.array(max_pct)
    #pct_dat = np.array(pcts) #what should this do?
    plt.figure()
    plt.plot(pct_dat.T)

    #print pct_dat.shape
    #print 
    avgs = pct_dat.mean(axis=1)
    plt.figure()
    #print clusts
    #print avgs
    plt.plot(clusts, avgs)


    plt.show()

    return counts, pct_dat


def analysis_logistic_regression(data):

    walking_data = data[data.act==4]
    #walking_data.reset_index(inplace=True)
    
    #X1 = X[:]
    scores = []
    for a in range(1,15):
        
        #dat = walking_data[walking_data.subj in [a,b]]
        X, y = make_freq_features(walking_data, delta=4)
        yi = np.copy(y)
        yi[yi != a] = 0
        Xi = X.iloc[:, 1:2]
        print Xi.shape
        #yi[yi == a+1] = 1


        X_train, X_test, y_train, y_test = train_test_split(Xi, yi, 
            train_size=.1, random_state=3)
        clf = LogisticRegression()
        
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        print score
        scores.append(score)
    print np.mean(scores)
    Xy = pd.DataFrame([Xi, yi])


def analysis_tree_win_size(data):
    walking_data = data[data.act==4]
    
    for ws in [3,4,5,6,7,10]:
        print 'Window Size:', ws
        X, y = make_time_features(walking_data, win_size=ws, jrk=1)
        print X.shape
        clf = tree.DecisionTreeClassifier()
        
        
        print 'Leave One User Out.'
        mn, sd = analysis_classify_walkers_louo(clf, X, y)
        print 'mean',mn, 'std', sd
        print
        print



def analysis_tree(X,y):
    clf = tree.DecisionTreeClassifier(min_samples_leaf=10)
    print X.shape
    
    #scores = cross_val_score(clf, X, y)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=.3, random_state=3)
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    print score
    print clf
    
    #print
    #print 'Leave One User Out.'
    #mn, sd = analysis_classify_walkers_louo(clf, X, y)
    #print 'mean',mn, 'std', sd
    


    return clf

def run_analyses(X, y):
    clf = tree.DecisionTreeClassifier(class_weight='balanced', max_depth=5, min_samples_leaf=5)#, min_samples_split=20)#, max_features=4)
    parameters = {
        'max_features': [3,4,5,6,7,8],
        'max_depth': [5,10,15,20],
        'min_samples_leaf': [3,4,5,6,7,8,9]}

    analysis_grid_tree(clf, parameters, X, y)

    print
    print

    clf = svm.SVC(gamma=1, class_weight='balanced')
    parameters = {
        'kernel': ('linear', 'rbf', 'sigmoid'), 
        'C': [0.1, 1., 10., 100.], 
        'gamma': [0.001, 0.1, 1., 10., 100.]}

    clf = analysis_grid_tree(clf, parameters, X, y)

    return clf


def analysis_grid_tree(clf, parameters, X, y):
    """
    """
    print 'data:', X.shape

    yi = np.copy(y)
     
    # revise y_labels for Leave One Out Analysis?
    yi[yi != 1] = 0
    #y_test[y_test != lo] = 0
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, yi, train_size=.6, random_state=3)#, stratify=yi)
    print 'train:', X_train.shape
    print 'test:', X_test.shape
    
    # Fit
    t = time.time()
    clf.fit(X_train, y_train)
    print 'Fit time:', time.time() - t

    # Predict
    t = time.time()
    clf.predict(X_test)
    print 'Predict time:', time.time() - t

    # Score
    score = clf.score(X_test, y_test)
    print 'Initial Classifier score:', score

    print 'untuned louo'
    scores = analysis_classify_walkers_louo(clf, X, y)
    print scores
    print scores.mean(), scores.std()

    print '* Grid Search *'
    grid = grid_search.GridSearchCV(clf, parameters)
    # Fit tuned
    t = time.time()
    grid.fit(X, yi)
    print 'Fit time:', time.time() - t

    # Predict tuned
    t = time.time()
    grid.predict(X_test)
    print 'Predict time', time.time() - t 

    # Score tuned
    print 'grid search'
    print grid.best_score_
    print grid.best_estimator_


    print
    print 'tuned louo'
    scores = analysis_classify_walkers_louo(clf, X, y, parms=grid.best_params_)
    print scores 
    print scores.mean(), scores.std()

    return clf #scores


def compare_time_freq(data):
    pt = 0
    clf = tree.DecisionTreeClassifier(class_weight='balanced', min_samples_leaf=10)#, min_samples_split=20)#, max_features=4)
    parameters = {
        'max_features':[3,4,5],
        #'max_depth':[None, 2,3,4],
        'min_samples_leaf':[1,2,3,4,5,10],
        'min_samples_split':[2,3,4,5,10,15,20]}

    
    datawalk = data[data.act==4]
    Xf, yf = make_freq_features(datawalk, delta=4)
    print ''
    print ' * GridTreee Freq factors'
    scores_f = analysis_grid_tree(clf, parameters, Xf, yf)

    Xt, yt = make_time_features(datawalk, win_size=4.923077, delta=40)   
    print ''
    print ' * GridTreee Time factors'
    scores_t = analysis_grid_tree(clf, parameters, Xt, yt)

    #print 'T-test comparing validation using time and frequency features'
    #print ttest_ind(scores_f, scores_t)
    #print scores_f, scores_t
    #return scores_f, scores_t 

    # PCA freq
    pca = PCA(n_components=18)
    Xfp = pca.fit_transform(Xf)
    if pt:
        plt.figure()
        plt.title('PCA freq features')
        plt.plot(range(1,19), np.cumsum(pca.explained_variance_ratio_), 'o-')
        plt.xlim(1,18)
        plt.show()    

    # PCA time
    pca = PCA(n_components=36)
    Xtp = pca.fit_transform(Xt)
    if pt:
        plt.figure()
        plt.title('PCA time features')
        plt.plot(range(1,37), np.cumsum(pca.explained_variance_ratio_), 'o-')
        plt.xlim(1,18)
        plt.show()  

    # PCA combinded
    Xall = np.hstack((Xf, Xt))
    print ''
    print ''
    
    print 'combined features', Xall.shape
    pca = PCA(n_components=54)
    Xallp = pca.fit_transform(Xall)
    if pt:
        plt.figure()
        plt.title('PCA freq and time features')
        plt.plot(range(1,55), np.cumsum(pca.explained_variance_ratio_), 'o-')
        plt.xlim(1,54)
        plt.show()  

    pca = PCA(n_components=5)
    Xallp = pca.fit_transform(Xall)
    print ''
    print ' * GridTreee combined factors'
    scores_a = analysis_grid_tree(clf, parameters, Xallp, yt) # labels are same
    print "scores freq"
    print scores_f
    print "scores time"
    print scores_t
    print "scores all"
    print scores_a

    p = pd.DataFrame(Xallp, columns=['PCA '+str(i) for i in range(5)])
    #p['y'] = y
    #print p.head()
    p.iloc[:,:5].hist(layout=(1,5),  figsize=(9,3))

    plt.figure()
    scatter_matrix(p, alpha=0.2, figsize=(6, 6), diagonal='kde')
    plt.show()



def analysis_freq_tree(data):
    walking_data = data[data.act==4]
    clf = tree.DecisionTreeClassifier()
    X, y = make_freq_features(walking_data)

    
    print 'Leave One User Out.'
    scores = analysis_classify_walkers_louo(clf, X, y)
    print 'mean',mn, 'std', sd





''' activity classification'''

def analysis_activity_time_tree(data):
    user_data = data[data.subj==4]
    clf = tree.DecisionTreeClassifier()
    
    X, y = make_time_features(user_data, ycol=act, yrng=[2,4,6])
    
    print 'Leave One User Out.'
    mn, sd = analysis_classify_walkers_louo(clf, X, y)
    print 'mean',mn, 'std', sd

    X, y = make
      
      
def analysis_activity_freq_tree(data):
    walking_data = data[data.act==4]
    clf = tree.DecisionTreeClassifier()
    X, y = make_freq_features(walking_data)

    print 'Leave One User Out.'
    mn, sd = analysis_classify_walkers_louo(clf, X, y)
    print 'mean',mn, 'std', sd




'''
Plots
'''

def time_domain_viz(data_files):
    subj = [1,5,8]
    t1,t2 = 520,1040

    dat = load_file(data_files[8], act=4)
    x = dat.ya[t1:t2]
    ts = dat.ts[t1:t2].as_matrix()
    r = calculate_ts_diffs(x, ts, delta=40, viz=1)


def exploratory_visualization(data_files):
    dat = load_file(data_files[0], act=4)
    dat.ts = dat.ts-min(dat.ts)
    #x = data.ya.as_matrix()

    if 0:
        #dat = data[data.act==4]
        plt.figure(figsize=(12,6))

        ax = plt.subplot(2,1,1)
        plt.plot(dat.ts, dat.ya, 'k')
        #pltsegs(ax, get_activity_segments(dat))
        ax.set_xlim([0, dat.ts.max()])
        plt.ylabel('Amplitude')

        ax = plt.subplot(2,1,2)
        plt.specgram(dat.ya, Fs=52., NFFT=256, noverlap=None)
        #pltsegs(ax, get_activity_segments(dat), labs=0)
        ax.set_ylim([0,26])
        ax.set_xlim([0, dat.ts.max()])
        plt.ylabel('Frequency')
        plt.xlabel('Time (s)')

        plt.tight_layout()
        plt.show()


    acc_3a(dat[2000:2520])
    plt.show()


def show_outliers(data_files):
    #using raw data
    datawalk = load_data(data_files, act=4, use_fix=False)

    #print datawalk.head()
    X, y = make_freq_features(datawalk, delta=40)

    pca = PCA(n_components=5)
    Xt = pca.fit_transform(X)

    p = pd.DataFrame(Xt)
    p['y'] = y
    #print p.head()
    p.iloc[:,:5].hist(layout=(1,5),  figsize=(9,3))

    # using fixed activity labels
    datawalk = load_data(data_files, act=4, use_fix=True)

    #print datawalk.head()
    X, y = make_freq_features(datawalk, delta=40)

    pca = PCA(n_components=5)
    Xt = pca.fit_transform(X)

    p = pd.DataFrame(Xt)
    p['y'] = y
    #print p.head()
    p.iloc[:,:5].hist(layout=(1,5),  figsize=(9,3))

    plt.show()
    return p

def show_misalignment(data_files):
    # pre alignment fix
    dat = load_file(data_files[6], use_fix=False)

    dat1 = dat[500*52:1501*52]
    dat1 = dat1.reset_index(drop=True)
    ns = dat1.shape[0]
    ts = np.linspace(0, ns/52., num=ns)

    plt.figure(figsize=(12,6))
    ax1 = plt.subplot(111)
    plt.plot(ts, dat1.xa, 'k')
    plt.title('Raw Acceleration Data')
    plt.ylabel('Acceleration')
    plt.xlabel('Time (s)')
    segs = get_activity_segments(dat1)
    pltsegs(ax1, segs)

    # use fixed activity labels
    dat = load_file(data_files[6])

    dat1 = dat[500*52:1501*52]
    dat1 = dat1.reset_index(drop=True)
    ns = dat1.shape[0]
    ts = np.linspace(0, ns/52., num=ns)

    plt.figure(figsize=(12,6))
    ax2 = plt.subplot(111)
    plt.plot(ts, dat1.xa, 'k')
    plt.title('Raw Acceleration Data with Fixed Labels')
    plt.ylabel('Acceleration')
    plt.xlabel('Time (s)')
    segs = get_activity_segments(dat1)
    pltsegs(ax2, segs)

    # show aperiodic section
    dat1 = dat[46000:47000]
    dat1 = dat1.reset_index(drop=True)
    ns = dat1.shape[0]
    ts = np.linspace(0, ns/52., num=ns)

    plt.figure(figsize=(12,6))
    ax = plt.subplot(111)
    plt.plot(ts, dat1.xa, 'k')
    plt.title('Raw Acceleration Data')
    plt.ylabel('Acceleration')
    plt.xlabel('Time (s)')
    segs = get_activity_segments(dat1)
    pltsegs(ax, segs)

    plt.show()


def show_features_by_subject(data_files=data_files):
    datawalk = load_data(data_files, act=4, use_fix=False)

    #print datawalk.head()
    X, y = make_freq_features(datawalk, delta=40)

    pca = PCA(n_components=5)
    Xt = pca.fit_transform(X)

    p = pd.DataFrame(Xt)
    p['Subject'] = y

    axes = p.boxplot(by='Subject', layout=(1,5))

    for i in range(5):
        axes[i].set_title('PCA '+str(i))
        axes[i].set_xlabel('')

    plt.show()


if __name__=="__main__":
    pass
    #data = load_data(data_files)

    #compare_time_freq(data)
    


