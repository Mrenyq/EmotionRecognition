import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model, to_categorical
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from ContrastiveLearning.Benchmark.BenchModel import createCLModel


def show_imgs(imgs, row, col):
    plt.figure()
    for i, img in enumerate(imgs):
        plot_num = i + 1
        plt.subplot(row, col, plot_num)
        plt.axis("off")
        plt.imshow(img)
    plt.show()


# Define const
NUM_CLASSES = 10
OPTIMIZER = Adam()
EPOCHS = 100

# Import CIFAR10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
y_train = to_categorical(y_train, NUM_CLASSES)
y_test = to_categorical(y_test, NUM_CLASSES)
input_shape = x_train.shape[1:4]
# print(input_shape)

# Define models
input_tensor = Input(shape=input_shape)
h, z = createCLModel(input_tensor)
encoder = Model(input_tensor, h)
CL_model = Model(encoder.input, z)
CL_model.summary()
plot_model(CL_model, to_file="BenchModel.png", show_shapes=True)

# max_img_num = 9
# imgs = []
# for d in datagen.flow(x_train[1:2], batch_size=1):
#     imgs.append(image.array_to_img(d[0], scale=True))
#     if (len(imgs) % max_img_num) == 0:
#         break
# show_imgs(imgs, row=3, col=3)


# Define contrastive loss. x1, x2 are augmented minibatch.
def contrastiveLoss(x1, x2, model: Model, temperature):
    z1 = model(x1)
    z2 = model(x2)
    z = np.zeros((z1.shape[0]*2,) + z1.shape[1:])
    batch_size_2 = len(z)
    z[::2] = z1.numpy()
    z[1::2] = z2.numpy()

    sim = np.zeros((batch_size_2, batch_size_2))
    for i in range(batch_size_2):
        for j in range(batch_size_2):
            sim[i, j] = np.dot(z[i], z[j]) / np.sqrt(np.dot(z[i], z[i]) * np.dot(z[j], z[j]))

    loss = np.zeros((batch_size_2, batch_size_2))
    for i in range(batch_size_2):
        sum = 0
        for k in range(batch_size_2):
            if k != i:
                sum += np.exp(sim[i, k] / temperature)
        for j in range(batch_size_2):
            loss[i, j] = -np.log(np.exp(sim[i, j] / temperature) / sum)

    loss_value = 0
    for k in range(batch_size_2 // 2):
        loss_value += loss[2*k-1, 2*k] + loss[2*k, 2*k-1]
    loss_value /= batch_size_2

    return loss_value


def computeGradient(x1, x2, model: Model, temperature):
    with tf.GradientTape() as tape:
        loss_value = contrastiveLoss(x1, x2, model, temperature)
    return loss_value, tape.gradient(loss_value, model.trainable_variables)


# Data augmentation
datagen = ImageDataGenerator(fill_mode="constant",
                             width_shift_range=0.3,
                             height_shift_range=0.3,
                             zoom_range=[0.4, 0.9],
                             channel_shift_range=0.5)

