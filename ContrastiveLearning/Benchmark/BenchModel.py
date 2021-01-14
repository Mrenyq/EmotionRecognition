import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization, Dropout, ReLU
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.vgg16 import VGG16


def createCLModel(input_tensor):
    vgg16 = VGG16(include_top=False, weights=None, input_tensor=input_tensor, pooling="avg")
    h = vgg16.output
    x = h
    for units in [128, 128]:
        x = Dense(units=units)(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
    z = x
    return h, z


class ContrastiveLearningModel(tf.keras.Model):
    def __init__(self):
        super(ContrastiveLearningModel, self).__init__(self)

        # Head(Dense)
        self.dense1 = Dense(units=128)
        self.dense2 = Dense(units=128)

        # Batch normalization
        self.batch_norm1 = BatchNormalization()
        self.batch_norm2 = BatchNormalization()

        # Activation
        self.activation = ReLU()

    def call(self, inputs, training=False, mask=None):
        encoder = VGG16(include_top=False, weights=None, input_tensor=inputs, pooling="avg")
        h = encoder.output
        x = self.activation(self.batch_norm1(self.dense1(h)))
        z = self.activation(self.batch_norm1(self.dense1(x)))
        return h, z




