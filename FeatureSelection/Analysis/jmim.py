import numpy as np
import pandas as pd
from Libs.Utils import valArLevelToLabels
from FeatureSelection.mifs.mifs import MutualInformationFeatureSelector
from sklearn.preprocessing import StandardScaler
import glob

features = []
y_ar = []
y_val = []

data_path = "G:\\usr\\nishihara\\data\\Features\\Yamaha-Experiment (2020-10-26 - 2020-11-06)\\data\\*"
game_result = "\\*_gameResults.csv"
path_result = "results\\"

# load features data
print("Loading data...")
for count, folder in enumerate(glob.glob(data_path)):
    for subject in glob.glob(folder + "\\*-2020-*"):
        eeg_path = subject + "\\results\\eeg\\"
        eda_path = subject + "\\results\\eda\\"
        ppg_path = subject + "\\results\\ppg\\"
        resp_path = subject + "\\results\\resp\\"
        ecg_path = subject + "\\results\\ecg\\"
        ecg_resp_path = subject + "\\results\\ecg_resp\\"

        features_list = pd.read_csv(subject + "\\features_list.csv")
        features_list["Valence"] = features_list["Valence"].apply(valArLevelToLabels)
        features_list["Arousal"] = features_list["Arousal"].apply(valArLevelToLabels)
        for i in range(len(features_list)):
            filename = features_list.iloc[i]["Idx"]
            eda_features = np.load(eda_path + "eda_" + str(filename) + ".npy")
            ppg_features = np.load(ppg_path + "ppg_" + str(filename) + ".npy")
            resp_features = np.load(resp_path + "resp_" + str(filename) + ".npy")
            eeg_features = np.load(eeg_path + "eeg_" + str(filename) + ".npy")
            ecg_features = np.load(ecg_path + "ecg_" + str(filename) + ".npy")
            ecg_resp_features = np.load(ecg_resp_path + "ecg_resp_" + str(filename) + ".npy")

            concat_features = np.concatenate(
                [eda_features, ppg_features, resp_features, eeg_features, ecg_features, ecg_resp_features])
            if np.sum(np.isinf(concat_features)) == 0 & np.sum(np.isnan(concat_features)) == 0:
                # print(eda_features.shape)
                features.append(concat_features)
                y_ar.append(features_list.iloc[i]["Arousal"])
                y_val.append(features_list.iloc[i]["Valence"])
            else:
                print(subject + "_" + str(i))
    print("{}/{}".format(count + 1, len(glob.glob(data_path))) + "\r", end="")
print("Finish")

# concatenate features and normalize them
X = np.concatenate([features])
scaler = StandardScaler()
X_norm = scaler.fit_transform(X)

y_ar = np.array(y_ar)
y_val = np.array(y_val)

feature_selector = MutualInformationFeatureSelector(method='JMIM',
                                                    k=5,
                                                    n_features='auto',
                                                    categorical=True,
                                                    verbose=2)

# Analyze the relevance between features and arousal
print("Analysing Arousal...")
feature_selector.fit(X_norm, y_ar)
