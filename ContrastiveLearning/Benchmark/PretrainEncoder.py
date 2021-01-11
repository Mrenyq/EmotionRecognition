import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.metrics import Mean
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
BATCH_SIZE = 64
OPTIMIZER = Adam()
EPOCHS_PRIOR = 100
T = 5

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
# plot_model(CL_model, to_file="BenchModel.png", show_shapes=True)


# Define contrastive loss. x1, x2 are augmented minibatch.
# @tf.function
def contrastiveLoss(x1, x2, model: Model, temperature=5):
    z1 = model(x1)
    z2 = model(x2)
    z = np.zeros((z1.shape[0]*2,) + z1.shape[1:])
    batch_size_2 = len(z)
    z[::2] = z1
    z[1::2] = z2

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

    # loss = loss.astype("float32")
    loss_value = 0.0
    for k in range(batch_size_2 // 2):
        loss_value += loss[2*k, 2*k+1] + loss[2*k+1, 2*k]
    loss_value /= batch_size_2
    loss_value = tf.constant(loss_value, dtype="float32")

    return loss_value


# @tf.function
def computeGradient(x1, x2, model: Model, temperature=5):
    with tf.GradientTape() as tape:
        loss_value = contrastiveLoss(x1, x2, model, temperature)
    return loss_value, tape.gradient(loss_value, model.trainable_variables)


# Data augmentation
datagen = ImageDataGenerator(fill_mode="constant",
                             width_shift_range=0.3,
                             height_shift_range=0.3,
                             zoom_range=[0.4, 0.9],
                             channel_shift_range=0.5)

g1 = datagen.flow(x_train, batch_size=BATCH_SIZE, shuffle=False)
g2 = datagen.flow(x_train, batch_size=BATCH_SIZE, shuffle=False)

d1 = g1.next()
d2 = g2.next()
max_img_num = 10
imgs = []
for i in range(max_img_num):
    imgs.append(image.array_to_img(x_train[i], scale=True))
    imgs.append(image.array_to_img(d1[i], scale=True))
    imgs.append(image.array_to_img(d2[i], scale=True))

show_imgs(imgs, row=10, col=3)

for epoch in range(EPOCHS_PRIOR):
    epoch_loss_avg = Mean()

    for x1, x2 in zip(g1, g2):
        loss_value, grad = computeGradient(x1, x2, CL_model, temperature=T)
        OPTIMIZER.apply_gradients(zip(grad, CL_model.trainable_variables))
        epoch_loss_avg(loss_value)

    print("Epoch {}/{} Loss: {:.3f}".format(epoch+1, EPOCHS_PRIOR, epoch_loss_avg.result().numpy()))




