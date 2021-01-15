import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import Mean, CategoricalAccuracy, Precision, Recall
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam, RMSprop, Adamax
from tensorflow.python.keras.utils.vis_utils import plot_model
from ContrastiveLearning.Benchmark.BenchModel import createClassificationModel, createCLModel_Small

# Define const
NUM_CLASSES = 10
BATCH_SIZE = 256
OPTIMIZER = Adam()
EPOCHS = 100
VALIDATION_SPLIT = 0.2

# Import CIFAR10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
y_train = to_categorical(y_train, NUM_CLASSES)
y_test = to_categorical(y_test, NUM_CLASSES)
input_shape = x_train.shape[1:4]

# Split training and validation data
idx_split = int(x_train.shape[0] * VALIDATION_SPLIT)
x_train, x_val = np.split(x_train, [-idx_split])
y_train, y_val = np.split(y_train, [-idx_split])

# Define encoder and load weights
input_tensor = Input(shape=input_shape)
h, _ = createCLModel_Small(input_tensor)
encoder = Model(input_tensor, h)
encoder.load_weights("./encoder_param.hdf5")

# Define classification model
logits = createClassificationModel(encoder.output, NUM_CLASSES)
pred = tf.nn.softmax(logits)
classification_model = Model(encoder.input, pred)
classification_model.summary()
plot_model(classification_model, to_file="ClassificationModel.png", show_shapes=True)

# Training and validation
classification_model.compile(optimizer=OPTIMIZER, loss=CategoricalCrossentropy(), metrics=[CategoricalAccuracy()])
history = classification_model.fit(x_train, y_train,
                                   batch_size=BATCH_SIZE,
                                   epochs=EPOCHS,
                                   validation_data=(x_val, y_val),
                                   verbose=2)

# Evaluate model
loss_train, acc_train = classification_model.evaluate(x_train, y_train, batch_size=BATCH_SIZE, verbose=0)
loss_val, acc_val = classification_model.evaluate(x_val, y_val, batch_size=BATCH_SIZE, verbose=0)
loss_test, acc_test = classification_model.evaluate(x_test, y_test, batch_size=BATCH_SIZE, verbose=0)
print("-----------------------------Result---------------------------")
print("Train: Loss: {:.3f}, Accuracy: {:.3%}".format(loss_train, acc_train))
print("Validation: Loss: {:.3f}, Accuracy: {:.3%}".format(loss_val, acc_val))
print("Test: Loss: {:.3f}, Accuracy: {:.3%}".format(loss_test, acc_test))

# Plot result
loss_history_train = history.history["loss"]
loss_history_val = history.history["val_loss"]
acc_history_train = history.history["acc"]
acc_history_val = history.history["val_acc"]
plt.figure()
plt.subplot(1, 2, 1)
plt.plot(loss_history_train)
plt.plot(loss_history_val)
plt.title('Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'])
plt.subplot(1, 2, 2)
plt.plot(acc_history_train)
plt.plot(acc_history_val)
plt.title('Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'])
plt.show()
