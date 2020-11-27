import numpy as np
import pandas as pd
from Libs.Utils import valArLevelToLabels
from FeatureSelection.mifs.mifs import MutualInformationFeatureSelector

date = '2020-10-27'
subject = 'A6-2020-10-27'
path = 'G:\\usr\\nishihara\\data\\Features\\Yamaha-Experiment (2020-10-26 - 2020-11-06)\\data\\' \
       + date + '\\' + subject + '\\'
path_features_list = path + 'features_list.csv'
path_features_data = path + 'results\\'
features_list = pd.read_csv(path_features_list)
label_ar = features_list['Arousal'].apply(valArLevelToLabels)
label_val = features_list['Valence'].apply(valArLevelToLabels)

features = []
for i in features_list['Idx'].values:
       ecg_features = np.load(path_features_data + 'ecg\\ecg_' + str(i) + '.npy')
       eeg_features = np.load(path_features_data + 'eeg\\eeg_' + str(i) + '.npy')
       resp_features = np.load(path_features_data + 'resp\\resp_' + str(i) + '.npy')
       eda_features = np.load(path_features_data + 'eda\\eda_' + str(i) + '.npy')
       ppg_features = np.load(path_features_data + 'ppg\\ppg_' + str(i) + '.npy')
       ecg_resp_features = np.load(path_features_data + 'ecg_resp\\ecg_resp_' + str(i) + '.npy')

       concat_features = np.concatenate([ecg_features, eeg_features, resp_features, eda_features, ppg_features, ecg_resp_features])
       features.append(concat_features)
features = np.array(features)

feature_selector = MutualInformationFeatureSelector(
       method='JMIM',
       k=5,
       n_features='auto',
       categorical=True,
       verbose=2)

# Analyze the relevance between features and arousal
feature_selector.fit(features, label_ar)
