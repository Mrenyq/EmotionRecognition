import numpy as np
import pandas as pd
import FeatureSelection.mifs.mifs as mifs

date = '2020-10-27'
subject = 'A6-2020-10-27'
path = 'G:\\usr\\nishihara\\data\\Features\\Yamaha-Experiment (2020-10-26 - 2020-11-06)\\data\\' \
       + date + '\\' + subject + '\\'
path_features_list = path + 'features_list.csv'
path_features_data = path + 'results\\'
features_list = pd.read_csv(path_features_list)

f_ecg = np.array([])
f_eeg = []
f_resp = []
f_eda = []
f_ppg = []
f_ecg_resp = []
for i in features_list['Idx'].values:
       ecg_tmp = np.load(path_features_data + 'ecg\\ecg_' + str(i) + '.npy')
       eeg_tmp = np.load(path_features_data + 'eeg\\eeg_' + str(i) + '.npy')
       resp_tmp = np.load(path_features_data + 'resp\\resp_' + str(i) + '.npy')
       eda_tmp = np.load(path_features_data + 'eda\\eda_' + str(i) + '.npy')
       ppg_tmp = np.load(path_features_data + 'ppg\\ppg_' + str(i) + '.npy')
       ecg_resp_tmp = np.load(path_features_data + 'ecg_resp\\ecg_resp_' + str(i) + '.npy')

       f_ecg = np.append(f_ecg, ecg_tmp, axis=0)
       if i == features_list['Idx'].values[0]:
              f_ecg = f_ecg.reshape(-1, len(ecg_tmp))

pass
