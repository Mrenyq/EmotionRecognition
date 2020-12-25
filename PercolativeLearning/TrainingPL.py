import numpy as np
import pandas as pd
import glob
import tensorflow as tf
from PercolativeLearning.PLModel import PercolativeLearning
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy, Mean, Precision, Recall
from tensorflow.keras.utils import plot_model, to_categorical
from sklearn.preprocessing import StandardScaler
from Libs.Utils import arToLabels, valToLabels, arValMulLabels, TrainingHistory

OPTIMIZER = Adam()
BATCH_SIZE = 256
EPOCHS = 100
VALIDATION_SPLIT = 0.1
TEST_SPLIT = 0.1
NUM_CLASSES = 4

STRIDE = 0.1
eda_features = []
ppg_features = []
resp_features = []
eeg_features = []
ecg_features = []
ecg_resp_features = []
y_ar = []
y_val = []

data_path = "G:\\usr\\nishihara\\data\\Yamaha-Experiment\\data\\2020-*"

# load features data
print("Loading data...")
for count, folder in enumerate(glob.glob(data_path)):
    print("{}/{}".format(count + 1, len(glob.glob(data_path))) + "\r", end="")
    for subject in glob.glob(folder + "\\*-2020-*"):
        eeg_path = subject + "\\results_stride=" + str(STRIDE) + "\\EEG\\"
        eda_path = subject + "\\results_stride=" + str(STRIDE) + "\\eda\\"
        ppg_path = subject + "\\results_stride=" + str(STRIDE) + "\\ppg\\"
        resp_path = subject + "\\results_stride=" + str(STRIDE) + "\\Resp\\"
        ecg_path = subject + "\\results_stride=" + str(STRIDE) + "\\ECG\\"
        ecg_resp_path = subject + "\\results_stride=" + str(STRIDE) + "\\ECG_resp\\"

        features_list = pd.read_csv(subject + "\\features_list_" + str(STRIDE) + ".csv")
        features_list["Valence"] = features_list["Valence"].apply(valToLabels)
        features_list["Arousal"] = features_list["Arousal"].apply(arToLabels)
        for i in range(len(features_list)):
            filename = features_list.iloc[i]["Idx"]
            eda_features.append(np.load(eda_path + "eda_" + str(filename) + ".npy"))
            ppg_features.append(np.load(ppg_path + "ppg_" + str(filename) + ".npy"))
            resp_features.append(np.load(resp_path + "resp_" + str(filename) + ".npy"))
            eeg_features.append(np.load(eeg_path + "eeg_" + str(filename) + ".npy"))
            ecg_features.append(np.load(ecg_path + "ecg_" + str(filename) + ".npy"))
            ecg_resp_features.append(np.load(ecg_resp_path + "ecg_resp_" + str(filename) + ".npy"))
            y_ar.append(features_list.iloc[i]["Arousal"])
            y_val.append(features_list.iloc[i]["Valence"])

eda_features = np.array(eda_features)
ppg_features = np.array(ppg_features)
resp_features = np.array(resp_features)
eeg_features = np.array(eeg_features)
ecg_features = np.array(ecg_features)
ecg_resp_features = np.array(ecg_resp_features)
x_main = np.concatenate([ecg_features], axis=1)
x_aux = np.concatenate([eda_features, ppg_features, resp_features, ecg_resp_features, eeg_features], axis=1)
y = [arValMulLabels(ar, val) for ar, val in zip(y_ar, y_val)]
input_dim_main = x_main.shape[1]
input_dim_aux = x_aux.shape[1]
input_dim_alpha = 1
num_data = x_main.shape[0]

# Transform y to one-hot vector
y = to_categorical(y, NUM_CLASSES)

# Standardize x_main and x_aux
ss = StandardScaler()
x_main = ss.fit_transform(x_main)
x_aux = ss.fit_transform(x_aux)

# Split train, validation and test
x_main_train, x_main_val, x_main_test = np.split(x_main, [-int(num_data * (VALIDATION_SPLIT + TEST_SPLIT)),
                                                          -int(num_data * TEST_SPLIT)], axis=0)
x_aux_train, x_aux_val, x_aux_test = np.split(x_aux, [-int(num_data * (VALIDATION_SPLIT + TEST_SPLIT)),
                                                      -int(num_data * TEST_SPLIT)], axis=0)
y_train, y_val, y_test = np.split(y, [-int(num_data * (VALIDATION_SPLIT + TEST_SPLIT)), -int(num_data * TEST_SPLIT)],
                                  axis=0)

# Make dataset
train_dataset = tf.data.Dataset.from_tensor_slices((x_main_train, x_aux_train, y_train)).shuffle(num_data).batch(
    BATCH_SIZE)
validation_dataset = tf.data.Dataset.from_tensor_slices((x_main_val, x_aux_val, y_val)).shuffle(num_data).batch(
    BATCH_SIZE)
test_dataset = tf.data.Dataset.from_tensor_slices((x_main_test, x_aux_test, y_test)).shuffle(num_data).batch(BATCH_SIZE)

# Define Model
input_main = Input(shape=(input_dim_main,))
input_aux = Input(shape=input_dim_aux)
input_alpha = Input(shape=input_dim_alpha)
PL = PercolativeLearning(num_classes=NUM_CLASSES)

feature = PL.createPercNet(input_main, input_aux, input_alpha)
perc_network = Model(inputs=[input_main, input_aux, input_alpha], outputs=feature)
logit = PL.createIntNet(input=perc_network.output)
whole_network = Model(inputs=[input_main, input_aux, input_alpha], outputs=logit)
perc_network.compile(optimizer=OPTIMIZER)
whole_network.compile(optimizer=OPTIMIZER, loss=CategoricalCrossentropy(), metrics=CategoricalAccuracy())
perc_network.summary()
whole_network.summary()
# plot_model(perc_network, to_file="perc_model.png", show_shapes=True)
# plot_model(whole_network, to_file="whole_model.png", show_shapes=True)

loss_metrics = CategoricalCrossentropy(from_logits=True)


# Define loss and gradient
def computeLoss(model: Model, x, y_true):
    y_pred = model(x)
    return loss_metrics(y_true, y_pred)


def computeGradient(model: Model, x, y_true):
    with tf.GradientTape() as tape:
        loss_value = computeLoss(model, x, y_true)
    return loss_value, tape.gradient(loss_value, model.trainable_variables)


train_result = TrainingHistory()
validation_result = TrainingHistory()


pass
