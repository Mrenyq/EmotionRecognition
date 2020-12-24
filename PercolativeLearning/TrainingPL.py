from PercolativeLearning.PLModel import PercolativeLearning
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy

OPTIMIZER = Adam()
input_dim_main = 2467
input_dim_aux = 13
input_dim_alpha = 1
num_classes = 4

input_main = Input(shape=(input_dim_main,))
input_aux = Input(shape=input_dim_aux)
input_alpha = Input(shape=input_dim_alpha)

PL = PercolativeLearning(num_classes=num_classes)
feature = PL.createPercNet(input_main, input_aux, input_alpha)
logit = PL.createIntNet(input=feature)

whole_network = Model(inputs=[input_main, input_aux, input_alpha], outputs=logit)
whole_network.compile(optimizer=OPTIMIZER, loss=CategoricalCrossentropy(), metrics=CategoricalAccuracy())
whole_network.summary()
