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
    print("{}/{}".format(count + 1, len(glob.glob(data_path))) + "\r", end="")
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
                [eda_features, ppg_features, resp_features, ecg_features, ecg_resp_features, eeg_features])
            if np.sum(np.isinf(concat_features)) == 0 & np.sum(np.isnan(concat_features)) == 0:
                # print(eda_features.shape)
                # print(concat_features.shape)
                features.append(concat_features)
                y_ar.append(features_list.iloc[i]["Arousal"])
                y_val.append(features_list.iloc[i]["Valence"])
            else:
                print(subject + "_" + str(i))

print("Finish")
print("EDA Features:", eda_features.shape[0])
print("PPG Features:", ppg_features.shape[0])
print("Resp Features:", resp_features.shape[0])
print("ECG Features:", ecg_features.shape[0])
print("ECG Resp Features:", ecg_resp_features.shape[0])
print("EEG Features:", eeg_features.shape[0])

# concatenate features and normalize them
X = np.concatenate([features])
scaler = StandardScaler()
X_norm = scaler.fit_transform(X)

y_ar = np.array(y_ar)
y_val = np.array(y_val)

print("All Features;", X_norm.shape[1])
print("Number of Data:", X_norm.shape[0])

# Define Feature Selector
feature_selector = MutualInformationFeatureSelector(method='JMIM',
                                                    k=5,
                                                    n_features=X_norm.shape[1],
                                                    categorical=True,
                                                    verbose=2)

# Analyze arousal
print("Analyzing Arousal...")
feature_selector.fit(X_norm, y_ar)

# Analyze valence
print("Analyzing Valence...")
feature_selector.fit(X_norm, y_val)
pass
