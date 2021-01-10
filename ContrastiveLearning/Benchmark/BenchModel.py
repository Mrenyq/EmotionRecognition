from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization, Dropout, ReLU
from tensorflow.keras.applications.resnet50 import ResNet50


def createCLModel(input_tensor):
    resnet50 = ResNet50(include_top=False, weights=None, input_tensor=input_tensor, pooling="avg")
    h = resnet50.output
    x = h
    for units in [128, 128]:
        x = Dense(units=units)(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
    z = x
    return h, z

