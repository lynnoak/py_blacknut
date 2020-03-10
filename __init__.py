"""

Python 3.5.4 |Anaconda custom (64-bit)| (default, Aug 14 2017, 13:41:13)
[MSC v.1900 64 bit (AMD64)]

"""


import time
import pprint
from src.myML import *
from src.datasets import *
from src.imputation import *
from sklearn import preprocessing
from collections import Counter
from imblearn.over_sampling import SMOTE


"""
Set for the configuration
Y = ["stop_kind","status"] for data_error
Y = ["audio", "play", "video"] for data_feedbacks
imput = ["Listwise","SimpleImputer","IterativeImputer","KNNImputer"]
X_train = ["original","listwise","running"] "running" only for data_error
strategy = ["mean","median","most_frequent"]
"""
myconf = ["audio","KNNImputer"," "," "]
title_conf = myconf[0]+'_'+myconf[1]+'_'+myconf[2]+'_'+myconf[3]+'_'

#Loading Dataset
#X,Y_running,Y_stopkind,Y_status,Y_stopinfor = data_2000()
X, Y_audio, Y_play, Y_video =data_feedbacks()

#Finding the Listwise and Only-running data
X_original = X
index_listwise = [i for i in range(X[:,0].size) if sum(np.isnan(X[i, :])) == 0]
X_listwise = X[index_listwise, :]

#index_running = [i for i in range(Y_running.size) if Y_running[i] == 1]
#X_running = X[index_running, :]
X_running = X_listwise

#Configuration for trainning data of the imputation algorithms
X_train_conf = {"original":X_original,"listwise":X_listwise,"running":X_running}
X_train = X_train_conf.get(myconf[2],X_original)

#Configuration for the imputation algorithms
myIMP_conf = {"Listwise":X_listwise,
        "SimpleImputer":mySimImp(X_original,X_train,myconf[3]),
        "IterativeImputer":myIterImp(X_original,X_train),
        "KNNImputer":myKNNImp(X_original)}
X = myIMP_conf.get(myconf[1],X_listwise)

"""
Configuration for parameters: 
K: K for KNN algorithms
num_constraints: Number of the constraints for metric learning algorithms
scoring: Parameter fotolr the KNN performance measure,detail in the function ColKNNScore()
"""
K = 5
num_constraints = 500#(int(X.shape[0]/500)+1)*100
scoring='acc'
S = {}
#Y = Y_stopkind
Y = Y_audio


if myconf[1] == "Listwise":
    Y = Y[index_listwise]
    
index = [i for i in range(len(Y)) if Counter(Y)[Y[i]]>=6]
X = X[index,:]
Y = Y[index]
X,Y = SMOTE().fit_resample(X, Y)

#Euclidean distance
S['Euc.'] = ColKNNScore(X,Y,K,scoring = scoring,title="Euc.")
my3Dplot(X,Y,title_conf+'Euc.')

#PCA
X_normalize = normalize(X,axis=0)
pca = decomposition.PCA(n_components=0.9).fit(X_normalize)
X_pca = pca.transform(X_normalize)
print("Number of dimension after PCA:"+str(len(X_pca[0]))+ "\n")
if len(X_pca[0])<3:
    pca = decomposition.PCA(n_components=3).fit(X_normalize)
    X_pca = pca.transform(X_normalize)
S['PCA']  = ColKNNScore(X_pca,Y,K,scoring = scoring,title="PCA")
my3Dplot(X_pca,Y,title_conf+'PCA')

#Metric Learning
St, metricL = myMl(X,Y,K =K, scoring=scoring,num_constraints=num_constraints,title=title_conf)
S =dict(S,**St)

#Saving the result
saveout = sys.stdout
file = open('output.txt', 'a')
sys.stdout = file
print("\n\n\n-------------------\n")
print (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())+"\n")
print("Target Label: " + myconf[0] + "\n")
print("Imputation Processing: " + myconf[1] + "\n")
if myconf[2] != '':
    print("Imputation Train Set: " + myconf[2] + "\n")
if myconf[3] != '':
    print("Imputation Strategy: " + myconf[3] + "\n")
print("Number of constraints:"+str(num_constraints)+ "\n")
print("Number of dimension after PCA:"+str(len(X_pca[0]))+ "\n")
print("\n-------------------\n\n")
pprint.pprint(S, width=1)
print("\n\n\n-------------------\n\n")
file.close()
sys.stdout = saveout




