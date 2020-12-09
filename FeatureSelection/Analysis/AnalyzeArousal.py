import numpy as np
import pandas as pd
from Libs.Utils import valArLevelToLabels
from FeatureSelection.mifs.mifs import MutualInformationFeatureSelector
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed
import glob
import os


def fitJMIM(x, y, selector):
    selector.fit(x, y)


eda_features = []
ppg_features = []
resp_features = []
eeg_features = []
ecg_features = []
ecg_resp_features = []
y_ar = []

data_path = "G:\\usr\\nishihara\\data\\Yamaha-Experiment\\data\\*"
# data_path = "G:\\usr\\nishihara\\data\\Yamaha-Experiment\\data\\2020-11-03"

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
        # features_list["Valence"] = features_list["Valence"].apply(valArLevelToLabels)
        features_list["Arousal"] = features_list["Arousal"].apply(valArLevelToLabels)
        for i in range(len(features_list)):
            filename = features_list.iloc[i]["Idx"]
            eda_features.append(np.load(eda_path + "eda_" + str(filename) + ".npy"))
            ppg_features.append(np.load(ppg_path + "ppg_" + str(filename) + ".npy"))
            resp_features.append(np.load(resp_path + "resp_" + str(filename) + ".npy"))
            eeg_features.append(np.load(eeg_path + "eeg_" + str(filename) + ".npy"))
            ecg_features.append(np.load(ecg_path + "ecg_" + str(filename) + ".npy"))
            ecg_resp_features.append(np.load(ecg_resp_path + "ecg_resp_" + str(filename) + ".npy"))
            y_ar.append(features_list.iloc[i]["Arousal"])

            # concat_features = np.concatenate(
            #     [eda_features, ppg_features, resp_features, ecg_features, ecg_resp_features, eeg_features])
            # if np.sum(np.isinf(concat_features)) == 0 & np.sum(np.isnan(concat_features)) == 0:
            #     # print(eda_features.shape)
            #     # print(concat_features.shape)
            #     features.append(concat_features)
            #     y_ar.append(features_list.iloc[i]["Arousal"])
            #     y_val.append(features_list.iloc[i]["Valence"])
            # else:
            #     print(subject + "_" + str(i))

# subject = data_path + "\\B3-2020-11-03"
# eeg_path = subject + "\\results\\eeg\\"
# eda_path = subject + "\\results\\eda\\"
# ppg_path = subject + "\\results\\ppg\\"
# resp_path = subject + "\\results\\resp\\"
# ecg_path = subject + "\\results\\ecg\\"
# ecg_resp_path = subject + "\\results\\ecg_resp\\"
#
# features_list = pd.read_csv(subject + "\\features_list.csv")
# features_list["Valence"] = features_list["Valence"].apply(valArLevelToLabels)
# features_list["Arousal"] = features_list["Arousal"].apply(valArLevelToLabels)
# for i in range(len(features_list)):
#     filename = features_list.iloc[i]["Idx"]
#     eda_features = np.load(eda_path + "eda_" + str(filename) + ".npy")
#     ppg_features = np.load(ppg_path + "ppg_" + str(filename) + ".npy")
#     resp_features = np.load(resp_path + "resp_" + str(filename) + ".npy")
#     eeg_features = np.load(eeg_path + "eeg_" + str(filename) + ".npy")
#     ecg_features = np.load(ecg_path + "ecg_" + str(filename) + ".npy")
#     ecg_resp_features = np.load(ecg_resp_path + "ecg_resp_" + str(filename) + ".npy")
#
#     concat_features = np.concatenate(
#         [eda_features, ppg_features, resp_features, ecg_features, ecg_resp_features, eeg_features])
#     if np.sum(np.isinf(concat_features)) == 0 & np.sum(np.isnan(concat_features)) == 0:
#         # print(eda_features.shape)
#         # print(concat_features.shape)
#         features.append(concat_features)
#         y_ar.append(features_list.iloc[i]["Arousal"])
#         y_val.append(features_list.iloc[i]["Valence"])
#     else:
#         print(subject + "_" + str(i))

features = []
features.append(np.array(eda_features))
features.append(np.array(ppg_features))
features.append(np.array(resp_features))
features.append(np.array(ecg_features))
features.append(np.array(ecg_resp_features))
features.append(np.array(eeg_features))

print("Finish")
print("EDA Features:", eda_features[0].shape[0])
print("PPG Features:", ppg_features[0].shape[0])
print("Resp Features:", resp_features[0].shape[0])
print("ECG Features:", ecg_features[0].shape[0])
print("ECG Resp Features:", ecg_resp_features[0].shape[0])
print("EEG Features:", eeg_features[0].shape[0])

# normalize features
scaler = StandardScaler()
len_features = 0
for i, feature in enumerate(features):
    features[i] = scaler.fit_transform(feature)
    len_features += feature.shape[1]  # count number of data
len_data = features[0].shape[0]

y_ar = np.array(y_ar)

print("All Features;", len_features)
print("Number of Data:", len_data)

# Define Feature Selector
feature_selector = []
for f, i in zip(features, [10, 5, 5, 5, 5, 10]):
    selector = MutualInformationFeatureSelector(method='JMIM',
                                                k=5,
                                                n_features=i,
                                                categorical=True,
                                                n_jobs=-1,
                                                verbose=2)
    feature_selector.append(selector)

# Analyze features
print("Analyzing Features...")
Parallel(n_jobs=-1)([delayed(fitJMIM)(x, y_ar, s) for x, s in zip(features, feature_selector)])

# Save result
path_result = "G:\\usr\\nishihara\\data\\Yamaha-Experiment\\jmim_results\\"
os.makedirs(path_result, exist_ok=True)
# print(feature_selector.ranking_, feature_selector.mi_)
results_jmi = []
results_ranking = []
for i, s in enumerate(feature_selector):
    result = np.zeros(len(s.mi_))
    result[s.ranking_] = s.mi_
    results_jmi.append(result)
    results_ranking.append(s.ranking_)

result_df = pd.DataFrame(
    {"JMI_EDA": results_jmi[0], "JMI_PPG": results_jmi[1], "JMI_Resp": results_jmi[2], "JMI_ECG": results_jmi[3],
     "JMI_ECG_Resp": results_jmi[4], "JMI_EEG": results_jmi[5],
     "Ranking_EDA": results_ranking[0], "Ranking_PPG": results_ranking[1], "Ranking_Resp": results_ranking[2],
     "Ranking_ECG": results_ranking[3], "Ranking_ECG_Resp": results_ranking[4], "Ranking_EEG": results_ranking[5]})

result_df.to_csv(path_result + "jmim_feature_analysis_ar.csv", index=False)
