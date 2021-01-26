import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, GlobalAveragePooling2D, BatchNormalization, Dropout, ReLU
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


def createCLModel_Small(input_tensor, num_output):
    x = input_tensor
    for filters in [64, 128, 256, 512]:
        x = Conv2D(filters, kernel_size=(3, 3), padding="same")(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
    h = GlobalAveragePooling2D()(x)

    x = h
    for units in [128, 128]:
        x = Dense(units=units)(x)
        x = ReLU()(x)
        x = Dropout(0.2)(x)
    z = Dense(units=num_output)(x)

    return h, z


def createClassificationModel(input_tensor, num_classes):
    x = input_tensor
    for u in [512, 256, 128]:
        x = Dense(units=u)(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
    logits = Dense(units=num_classes)(x)
    return logits


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




