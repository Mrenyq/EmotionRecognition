from KnowledgeDistillation.Models.EnsembleDistillModel import EnsembleStudent
import pandas as pd
import numpy as np
import glob

path = "G:\\usr\\nishihara\\data\\Yamaha-Experiment\\data\\2020-10-27\\A6-2020-10-27\\"
path_ecgraw = path + "results\\ecg_raw\\"
ecg_raw = []
ecg_len = []

features_list = pd.read_csv(path + "features_list.csv")
idx = features_list["Idx"].values

for i in idx:
    load_data = np.load(path_ecgraw + "ecg_raw_" + str(i) + ".npy")
    if len(load_data) >= 11700:
        ecg_len.append(len(load_data))
        ecg_raw.append(load_data)

len_min = np.nanmin(ecg_len)
for i, ecg in enumerate(ecg_raw):
    if len(ecg) > len_min:
        ecg_raw[i] = np.delete(ecg, np.s_[len_min:], axis=0)


