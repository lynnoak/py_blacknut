"""

Python 3.5.4 |Anaconda custom (64-bit)| (default, Aug 14 2017, 13:41:13)
[MSC v.1900 64 bit (AMD64)]

"""


import pprint
from src.datasets import *
from src.myMLtools import *
from src.myImputationTools import *
from src.submodular_imputation import *

"""
Set for the configuration
Y = ["stop_kind","status"] for data_error
Y = ["audio", "play", "video"] for data_feedbacks
imput = ['Listwise','SimpleImputer_mean','SimpleImputer_median','SimpleImputer_most_frequent','IterativeImputer','KNNImputer']
X_train = ["original","listwise","running"] "running" only for data_error
"""

myconf = ["audio","KNNImputer"," ",]
title_conf = myconf[0]+'_'+myconf[1]+'_'+myconf[2]+'_'

#Loading Dataset
#X,Y_running,Y_stopkind,Y_status,Y_stopinfor = data_2000()
X, Y_audio, Y_play, Y_video =data_feedbacks()
#Y = Y_stopkind
y = Y_audio

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

imp = Imputer(X = X,y =y,X_train=X_train,alg = myconf[1],metric = 'nan_euclidean')
X_imp = imp.X_imp[myconf[1]]
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

#Metric Learning
ml = NormalML(alg=['NCA'], X=X_imp, y=y, num_constraints=num_constraints,n_neighbors = K ,P_power = 2, scoring = scoring)
S = ml.ColKNNScore()

submodular = SubmodularWithNan(X_nan = X, y=y, imp_alg = myconf[1],style = 0, num_constraints =num_constraints)
St = submodular.ColKNNScore()
S = dict(S,**St)

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
print("\n-------------------\n\n")
pprint.pprint(S, width=1)
print("\n\n\n-------------------\n\n")
file.close()
sys.stdout = saveout




