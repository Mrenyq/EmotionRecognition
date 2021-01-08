from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model, to_categorical
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from ContrastiveLearning.Benchmark.BenchModel import createCLModel

# Define const
NUM_CLASSES = 10

# Import CIFAR10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
y_train = to_categorical(y_train, NUM_CLASSES)
y_test = to_categorical(y_test, NUM_CLASSES)

input_tensor = Input(shape=(32, 32, 3))
logits = createCLModel(input_tensor, num_output=4)
model = Model(input_tensor, logits)
model.summary()
plot_model(model, to_file="BenchModel.png", show_shapes=True)

