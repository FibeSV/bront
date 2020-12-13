import warnings
warnings.filterwarnings('ignore')
import librosa
import numpy as np
import IPython.display as ipd
import noisereduce as nr
import matplotlib.pyplot as plt
import librosa.display
import python_speech_features
from python_speech_features import mfcc
import sklearn
import os
#для нейронных сетей
import keras
import tensorflow as tf
from keras import layers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten,LSTM,TimeDistributed
from keras.layers import Convolution2D, MaxPooling2D,MaxPooling1D,Conv2D, AveragePooling2D
from keras.optimizers import Adam,SGD
from keras.utils import np_utils
from sklearn import metrics
import random
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import optimizers
import datetime
import sklearn
from sklearn.model_selection import train_test_split  
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import auc
class Bront:
    def __init__(self ,ww = None,hop= None, ww_mfc=2048, hop_mfc=512,sr= 22050,
                 n_input=20, n_output=0, model = None, path_main = '',
                 audio_flder ='audio',marks='marks',models='models', feat = 'feat_data', dta_f ='maindata'):
        #путь к папке с данными  
        self.main_folder_path = path_main 
        self.audio_flder = self.main_folder_path + audio + '/'
        self.marks_folder = self.main_folder_path + marks + '/'
        self.
        self.models_folder = self.main_folder_path + models + '/'
        self.feat_data_folder = self.main_folder_path + feat + '/'
        self.main_data = self.main_folder_path + dta_f + '/'
        #настройки кадрирования данных
        self.window_wide = ww
        self.hop = hop
        #настройки кадрирования для выделения мел-кепстральных коэффициентов
        self.window_wide_mfc = ww_mfc
        self.hop_mfc = hop_mfc
        #колличество входящих и выходящих данных
        self.n_input = n_input
        self.n_output = n_output
        #прочие 
        self.sr = sr
        #изучаемая модель 
        self.model = model
        
    ########------settings------##########    
    def set_wh(self, wide_wind, hop):
        self.window_wide = wide_wind
        self.hop = hop
    
    def set_mfcc(self, ww_mfc, hop_mfc):
        self.window_wide_mfc = ww_mfc
        self.hop_mfc = hop_mfc
        
    def set_main_path(self, path):
        self.main_folder_path = path
    
    ########-------info-------###########
    def main_folder_info(self):
        print(os.listdir(self.main_folder_path))
    
    #######---work whith data----########
    def preparation_for_edu(self, train_path=''):
        self.feature_ex_and_safe(self.audio_flder,self.feat_data_folder)
        self.data_maker(self.feat_data_folder, self.main_data)
        self.mark_ex_and_save(self.marks_folder, self.)
        
        
    def data_maker(self, path, path_save):
        files = np.array(os.listdir(path))
        for file in files:
            features = np.load(path+file)
            np.save(path_save+file.split('.')[0]+"__data.npy", self._data_ex(features))
            print(path_save+file.split('.')[0]+"__data.npy")    
    
    
    def _data_ex(self, features):
        rng = range(0, features.shape[1], self.hop)
        dat= np.zeros([len(rng), self.n_input, self.window_wide])
        for i,step in enumerate(rng):
            try:
                dat[i] = features[:,step:step+self.window_wide]
            except:
                pass
        return dat
    
    def _mfcc(self, data):
        m_htk = librosa.feature.mfcc(y=data, sr=self.sr, hop_length=self.hop_mfc,win_length=self.window_wide_mfc, n_mfcc=self.n_input)
        m_htk = sklearn.preprocessing.scale(m_htk, axis=1)
       #mfcc(data,sr,1/sr*(2*self.hop),1/sr*(2*self.hop), numcep = n, nfft = int(sr*self.window_wide) )
        return m_htk
    
    def feature_ex_and_safe(self, path, path_save):
        files = np.array(os.listdir(path))
        for file in files:
            x , sr = librosa.load(path+file)
            np.save(path_save+file.split('.')[0]+"__mfcc.npy",self._mfcc(x[:]))
            print(path_save+file.split('.')[0]+"__mfcc.npy")    
            
    def mark_ex_and_save(self, path, path_save):
        files = np.array(os.listdir(path))
        for file in files:
            start, end = self._get_mark(path+file)
            steps = range(0,end[-1], self.hop*self.hop_mfc)
            target = np.zeros(len(steps))
            k=0
            for i,step in enumerate(steps): 
                if step>end[k]:
                    k+=1
                    if k>=end.shape[0]:
                        break
                if (step<start[k]) and (end[k]<step+self.window_wide*self.hop_mfc): 
                    target[i]=1
            np.save(path_save+file.split('.')[0]+"__target.npy",target)
            print(path_save+file.split('.')[0]+"__target.npy")
            
    def _get_mark(self, path):
        file = open(path)
        raw_mark = file.read(-1)    
        
        start = np.zeros(len(raw_mark.split('\t\n'))-1)
        end = np.zeros(len(raw_mark.split('\t\n'))-1)
        
        for i,mark in enumerate(raw_mark.split('\t\n')):
            if len(mark)==0:
                continue
            try :
                start[i], end[i] = mark.split('\t')
            except:
                try:
                    start[i], end[i] = mark.split('\t')[1].split("\n")[1],mark.split('\t')[2]
                except:
                    pass
        start = np.array(start*self.sr, dtype="int")
        end = np.array(end*self.sr, dtype="int")
        return start, end
            
            
    def _wavelet(self, data):
        pass
