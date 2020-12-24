from tensorflow.keras.layers import Dense, Activation, BatchNormalization, Concatenate, Multiply


class PercolativeLearning():

    def __init__(self, num_classes):
        self.num_classes = num_classes

    def createPercNet(self, input_main, input_aux, alpha):
        x_main = input_main
        x_aux = Multiply()([input_aux, alpha])
        x = Concatenate()([x_main, x_aux])

        for u in [1024, 512, 256, 128]:
            x = Dense(units=u)(x)
            x = BatchNormalization()(x)
            x = Activation("relu")(x)

        return x

    def createIntNet(self, input):
        x = input
        for u in [128, 64, 32]:
            x = Dense(units=u)(x)
            x = BatchNormalization()(x)
            x = Activation("relu")(x)

        logit = Dense(units=self.num_classes)(x)
        return logit
