import sys
sys.path.append("..")
import numpy as np
from sklearn import preprocessing
from src.imputation import *

localrep ="./data/"

def data_feedbacks():
    """
    Dataset from the error information
    X :features,
    Y_running : the "running_at_$date" with value [0,1]
    Y_stopkind : the "stop_kind" with value {'game','client_disc','conn_to','streamer','stop','stale','ghost','client_stopped','relaunch','cancel','streamer_to'}
    Y_status : the "status" with value {'ended', 'error', 'cancelled'}
    Y_stopinfor : the binarization of the "serves_stop_kind" with the feature 'user_exit','client','no error','other	server','streamer'
    """
    file = localrep + "data_feedbacks.csv"
    X_file = np.genfromtxt(file,skip_header=1, filling_values=np.nan, delimiter=",")
    X = X_file[:, 4:-2]
    Y_stopkind = np.genfromtxt(file, skip_header=1, dtype='str', usecols=(-2), delimiter=",")
    lb = preprocessing.LabelBinarizer()
    lb.fit(Y_stopkind)
    Y_stopkind= lb.transform(Y_stopkind)
    Y_status = np.genfromtxt(file,skip_header=1, dtype = 'str',usecols = (-1), delimiter=",")
    lb = preprocessing.LabelBinarizer()
    lb.fit(Y_status)
    Y_status= lb.transform(Y_status)
    X = np.column_stack((X,Y_stopkind,Y_status))

    Y_audio = X_file[:,1]
    Y_play = X_file[:,2]
    Y_video = X_file[:,3]

    return X, Y_audio, Y_play, Y_video


def data_error():
    """
    Dataset from the error information
    X :features,
    Y_running : the "running_at_$date" with value [0,1]
    Y_stopkind : the "stop_kind" with value {'game','client_disc','conn_to','streamer','stop','stale','ghost','client_stopped','relaunch','cancel','streamer_to'}
    Y_status : the "status" with value {'ended', 'error', 'cancelled'}
    Y_stopinfor : the binarization of the "serves_stop_kind" with the feature 'user_exit','client','no error','other	server','streamer'
    """
    file = localrep + "data_error.csv"
    X_file = np.genfromtxt(file,skip_header=1, filling_values=np.nan, delimiter=",")
    X = X_file[:, :-10]
    Y_running = X_file[:, -9]

    Y_stopkind = np.genfromtxt(file,skip_header=1, dtype = 'str',usecols = (-8), delimiter=",")
    le = preprocessing.LabelEncoder()
    le.fit(Y_stopkind)
    Y_stopkind = le.transform(Y_stopkind)
    Y_status = np.genfromtxt(file,skip_header=1, dtype = 'str',usecols = (-7), delimiter=",")
    le = preprocessing.LabelEncoder()
    le.fit(Y_status)
    Y_status = le.transform(Y_status)
    Y_stopinfor = X_file[:, -6:]

    return X,Y_running,Y_stopkind,Y_status,Y_stopinfor

def data_2000():
    """
    Sub set of the dataset
    """
    [X, Y_running, Y_stopkind, Y_status, Y_stopinfor] = data_error()

    return X[:2000,:],Y_running[:2000],Y_stopkind[:2000],Y_status[:2000],Y_stopinfor[:2000]

"""
data = ["2000","error","feedback"]
Y = ["stop_kind","status"] for data_error
Y = ["audio", "play", "video"] for data_feedbacks
imput = ["Listwise","SimpleImputer","IterativeImputer","KNNImputer"]
X_train = ["original","listwise","running"] "running" only for data_error
strategy = ["mean","median","most_frequent"]

def data_imputation(data = "2000",Y ="stop_kind",imput ="Listwise",X_train ="original",strategy ="mean"):
    if data == "feedback":
        X, Y_audio, Y_play, Y_video =data_feedbacks()
        Y_conf = {"audio":Y_audio, "play":Y_play, "video":Y_video}
        Y = Y_conf.get(Y,Y_audio)
    else:
        X, Y_running, Y_stopkind, Y_status, Y_stopinfor = data_2000()
        Y_conf = {"stopkind":Y_stopkind, "status":Y_status}# "stopinfor":Y_stopinfor}
        Y = Y_conf.get(Y,Y_stopkind)
        index_running = [i for i in range(Y_running.size) if Y_running[i] == 1]
        X_running = X[index_running, :]

    # Finding the Listwise and Only-running data
    X_original = X
    index_listwise = [i for i in range(X[:, 0].size) if sum(np.isnan(X[i, :])) == 0]
    X_listwise = X[index_listwise, :]


    # Configuration for trainning data of the imputation algorithms
    X_train_conf = {"original": X_original, "listwise": X_listwise, "running": X_running}
    X_train = X_train_conf.get(X_train, X_original)

    # Configuration for the imputation algorithms
    myIMP_conf = {"Listwise": X_listwise,
                  "SimpleImputer": mySimImp(X_original, X_train, strategy),
                  "IterativeImputer": myIterImp(X_original, X_train),
                  "KNNImputer": myKNNImp(X_original)}
    X = myIMP_conf.get(imput, X_listwise)
    if imput == "Listwise":
        Y = Y[index_listwise]

    return X,Y

S = {}
S['error_stop_kind_Listwise'] = data_imputation()
S['error_stop_kind_SimpleImputer_orginal_mean'] =data_imputation(imput="SimpleImputer")
S['error_stop_kind_SimpleImputer_listwise_mean'] =data_imputation(imput="SimpleImputer",X_train="listwise")
S['error_stop_kind_SimpleImputer_running_mean'] =data_imputation(imput="SimpleImputer",X_train="running")
S['error_stop_kind_SimpleImputer_orginal_median'] =data_imputation(imput="SimpleImputer",strategy="median")
S['error_stop_kind_SimpleImputer_listwise_median'] =data_imputation(imput="SimpleImputer",X_train="listwise",strategy="median")
S['error_stop_kind_SimpleImputer_running_median'] =data_imputation(imput="SimpleImputer",X_train="running",strategy="median")
S['error_stop_kind_SimpleImputer_orginal_most_frequent'] =data_imputation(imput="SimpleImputer",strategy="most_frequent")
S['error_stop_kind_SimpleImputer_listwise_most_frequent'] =data_imputation(imput="SimpleImputer",X_train="listwise",strategy="most_frequent")
S['error_stop_kind_SimpleImputer_running_meanmost_frequent'] =data_imputation(imput="SimpleImputer",X_train="running",strategy="most_frequent")
S['error_stop_kind_IterativeImputer_orginal'] =data_imputation(imput="IterativeImputer")
S['error_stop_kind_IterativeImputer_listwise'] =data_imputation(imput="IterativeImputer",X_train="listwise")
S['error_stop_kind_IterativeImputer_running'] =data_imputation(imput="IterativeImputer",X_train="running")
S['error_stop_kind_KNNImputer'] =data_imputation(imput="KNNImputer")


S['feedback_stop_kind_Listwise'] = data_imputation(data ="feedback")
S['feedback_stop_kind_SimpleImputer_orginal_mean'] =data_imputation(data ="feedback",imput="SimpleImputer")
S['feedback_stop_kind_SimpleImputer_listwise_mean'] =data_imputation(data ="feedback",imput="SimpleImputer",X_train="listwise")
S['feedback_stop_kind_SimpleImputer_running_mean'] =data_imputation(data ="feedback",imput="SimpleImputer",X_train="running")
S['feedback_stop_kind_SimpleImputer_orginal_median'] =data_imputation(data ="feedback",imput="SimpleImputer",strategy="median")
S['feedback_stop_kind_SimpleImputer_listwise_median'] =data_imputation(data ="feedback",imput="SimpleImputer",X_train="listwise",strategy="median")
S['feedback_stop_kind_SimpleImputer_running_median'] =data_imputation(data ="feedback",imput="SimpleImputer",X_train="running",strategy="median")
S['feedback_stop_kind_SimpleImputer_orginal_most_frequent'] =data_imputation(data ="feedback",imput="SimpleImputer",strategy="most_frequent")
S['feedback_stop_kind_SimpleImputer_listwise_most_frequent'] =data_imputation(data ="feedback",imput="SimpleImputer",X_train="listwise",strategy="most_frequent")
S['feedback_stop_kind_SimpleImputer_running_meanmost_frequent'] =data_imputation(data ="feedback",imput="SimpleImputer",X_train="running",strategy="most_frequent")
S['feedback_stop_kind_IterativeImputer_orginal'] =data_imputation(data ="feedback",imput="IterativeImputer")
S['feedback_stop_kind_IterativeImputer_listwise'] =data_imputation(data ="feedback",imput="IterativeImputer",X_train="listwise")
S['feedback_stop_kind_IterativeImputer_running'] =data_imputation(data ="feedback",imput="IterativeImputer",X_train="running")
S['feedback_stop_kind_KNNImputer'] =data_imputation(data ="feedback",imput="KNNImputer")

"""