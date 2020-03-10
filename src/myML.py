"""
@author: victor

metric computation for knn test

"""

import sys
sys.path.append("..")
import numpy as np

from sklearn import *
from sklearn.neighbors import *
from sklearn.metrics import *
from metric_learn import *
from collections import Counter

from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def ColKNNScore(X, Y, K=5, scoring='acc', title=""):
    """
    Compute the cross validation score of the KNN

    :param X: Data feature
    :param Y: Label
    :param K: K of KNN algorithm
    :param scoring: the configuration of the output,
                {'acc': accuracy rate,
                  'pre': precision score,
                  'rec': recall score,
                  'f_1': f1 score, }
    :param title: title information
    :return: S = (S_mea, S_std):the mean and std of KNNscore
    """
    S = {}

    ditscoring = {'acc': 'accuracy',
                  'pre': metrics.make_scorer(metrics.precision_score, average='weighted'),
                  'rec': metrics.make_scorer(metrics.recall_score, average='weighted'),
                  'f_1': metrics.make_scorer(metrics.f1_score, average='weighted')}

    if (scoring == 'test'):
        scoring = ['acc', 'f_1', 'pre', 'rec']

    if not isinstance(scoring, list):
        scoring = [scoring]

    for s in scoring:
        S_mea = []
        S_std = []
        for i in range(5):
            KNN = KNeighborsClassifier(n_neighbors=K)
            KNN.fit(X, Y)
            kf = model_selection.StratifiedKFold(n_splits=3, shuffle=True)
            score_KNN = model_selection.cross_val_score(KNN, X, Y, cv=kf,
                                                        scoring=ditscoring.get(s, 'accuracy'))
            S_mea.append(score_KNN.mean())
            S_std.append(score_KNN.std() * 2)
        S_mea = np.mean(S_mea)
        S_std = np.mean(S_std)
        print(title + " " + s + " : %0.4f(+/-)%0.4f " % (S_mea, S_std))
        S[s] = (S_mea, S_std)
    return S

def my3Dplot(X,Y,title = " "):
    return 0
    """
    Show the 3D vesion of the distance in the latent space

    :param X: Data feature
    :param Y: Label
    :param title: title information
    :return: the figure
    
    fig = plt.figure()
    le = preprocessing.LabelEncoder()
    le.fit(Y)
    Y = le.transform(Y)
    color = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    color = [color[int(i)%7] for i in Y]

    PCAK = 3
    if (len(X[0]) > PCAK):
        pca = PCA(n_components=PCAK)
        X = pca.fit_transform(X)
        X = normalize(X)

    ax = plt.subplot(111, projection='3d')
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=color)
    ax.set_title(title)

    plt.savefig(title)
    plt.show()
    """


def myMl(X, Y, K=5, scoring='acc', num_constraints =100, title=" "):
    """
    compute the metric learning result

    :param X: Data feature
    :param Y: Label
    :param K: K of KNN alrigothm
    :param scoring: the configuration of the output,
                {'acc': accuracy rate,
                  'pre': precision score,
                  'rec': recall score,
                  'f_1': f1 score, }
    :param num_constraints: number of the constraints for metric learning algorithms
    :param title: title information
    :return: S = (S_mea, S_std):the dictionary contains mean and std of KNNscore of every algorithms
        metricL: the dictionary contains the learned metric of every algorithms
    """
    S = {}
    metricL = {}


    try:

        nca = NCA()
        nca.fit(X, Y)
        XN = nca.transform(X)

        S_NCA = ColKNNScore(XN, Y, K, scoring=scoring, title="NCA")
        my3Dplot(XN, Y, title + 'NCA')
        metricL_NCA = nca.get_metric()
    except:
        S_NCA = 0
        metricL_NCA = 0
        print('NCA error')

    S['NCA'] = S_NCA
    metricL['NCA'] = metricL_NCA

    try:
        lfda = LFDA(k=3)
        try:
            lfda.fit(X, Y)
        except:
            tX = X + 10 ** -4
            lfda.fit(tX, Y)
        XLf = lfda.transform(X)
        index_del = [i for i in range(len(XLf[0])) if np.isnan(XLf[0, i]) != True]
        XLf = XLf[:,index_del]

        S_LFDA = ColKNNScore(XLf, Y, K, scoring=scoring, title="LFDA")
        my3Dplot(XLf, Y, title + 'LFDA')
        metricL_LFDA = lfda.get_metric()
    except:
        print('LFDA error')
        S_LFDA = 0
        metricL_LFDA = 0

    S['LFDA'] = S_LFDA
    metricL['LFDA'] =metricL_LFDA

    try:

        lsml = LSML_Supervised(num_constraints=num_constraints)
        try:
            lsml.fit(X, Y)
        except:
            tX = X + 10 ** -4
            lsml.fit(tX, Y)
        XLs = lsml.transform(X)

        S_LSML = ColKNNScore(XLs, Y, K, scoring=scoring, title="LSML")
        my3Dplot(XLs, Y, title + 'LSML')
        metricL_LSML = lsml.get_metric()
    except:
        print('LSML error')
        S_LSML = 0
        metricL_LSML = 0

    S['LSML'] = S_LSML
    metricL['LSML'] =metricL_LSML

    return S, metricL

"""
    try:
        lmnn = LMNN()
        try:
            lmnn.fit(X, Y)
        except:
            # remove the small classes
            class_del = np.where(np.bincount(Y) <= 5)
            index_del = [i for i in range(len(Y)) if np.all(Y[i]!=class_del[0])]
            Xd = X[index_del,:]
            Yd = Y[index_del]
            lmnn.fit(Xd, Yd)
        XL = lmnn.transform(X)

        S_LMNN = ColKNNScore(XL, Y, K, scoring=scoring, title="LMNN")
        my3Dplot(XL, Y, title + 'LMNN')
        metricL_LMNN = lmnn.get_metric()
    except:
        S_LMNN = 0
        metricL_LMNN = 0
        print('LMNN error')

    S['LMNN'] = S_LMNN
    metricL['LMNN'] = metricL_LMNN

    try:
        itml = ITML_Supervised(num_constraints=num_constraints)
        try:
            itml.fit(X, Y)
        except:
            tX = X + 10 ** -4
            itml.fit(X, Y)
        XI = itml.transform(X)

        S_ITML = ColKNNScore(XI, Y, K, scoring=scoring, title="ITML")
        my3Dplot(XI, Y, title + 'ITML')
        metricL_ITML = itml.get_metric()
    except:
        S_ITML = 0
        metricL_ITML =0
        print('ITML error')

    S['ITML'] = S_ITML
    metricL['ITML'] =  metricL_ITML


"""








