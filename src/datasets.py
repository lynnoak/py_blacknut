import sys
sys.path.append("..")
import numpy as np
from sklearn import preprocessing

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

