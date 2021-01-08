import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization, Dropout, ReLU
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers.experimental.preprocessing import RandomCrop, RandomContrast


def createCLModel(input_tensor):
    x = RandomCrop(25, 25)(input_tensor)
    augmented = RandomContrast(0.2)(x)
    resnet50 = ResNet50(include_top=False, weights=None, input_tensor=augmented, pooling="avg")
    h = resnet50.output
    x = h
    for units in [128, 128]:
        x = Dense(units=units)(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
    z = x
    return h, z

