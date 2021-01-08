import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization, Dropout, ReLU
from tensorflow.keras.applications.resnet50 import ResNet50


def createCLModel(input_tensor, num_output):
    resnet50 = ResNet50(include_top=False, weights=None, input_tensor=input_tensor, pooling="avg")
    x = resnet50.output
    for units in [128, 128]:
        x = Dense(units=units)(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)

    logits = Dense(units=num_output)(x)
    return logits

