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
from ContrastiveLearning.Benchmark.BenchModel import createCLModel, ContrastiveLearningModel


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
BATCH_SIZE = 128
OPTIMIZER = Adam()
EPOCHS_PRIOR = 100
T = 0.1

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
@tf.function
def contrastiveLoss(xis, xjs, model: Model, temperature=0.1):
    xis = tf.convert_to_tensor(xis, dtype=tf.float32)
    xjs = tf.convert_to_tensor(xjs, dtype=tf.float32)
    temperature = tf.convert_to_tensor(temperature, dtype=tf.float32)
    zis = model(xis)
    zjs = model(xjs)
    zis = tf.math.l2_normalize(zis, axis=1)
    zjs = tf.math.l2_normalize(zjs, axis=1)
    z_all = tf.concat([zis, zjs], axis=0)
    # print(z_all.shape)
    batch_size_2 = z_all.shape[0]

    prod = tf.matmul(z_all, z_all, transpose_b=True)
    norm = tf.sqrt(tf.reduce_sum(z_all * z_all, axis=1, keepdims=True))
    norm = tf.matmul(norm, norm, transpose_b=True)
    sim = tf.truediv(prod, norm)
    sim_diag = tf.linalg.diag_part(sim)

    sum = tf.reduce_sum(tf.exp(sim / temperature), axis=1)
    sum = sum - tf.exp(sim_diag / temperature)
    sum = tf.tile(tf.expand_dims(sum, axis=1), [1, batch_size_2])
    loss = -tf.math.log(tf.exp(sim / temperature) / sum)

    loss_value = tf.constant(0.0, dtype=tf.float32)
    for k in range(batch_size_2 // 2):
        loss_value = loss_value + loss[k, k + (batch_size_2 // 2)] + loss[k + (batch_size_2 // 2), k]
    loss_value /= batch_size_2

    return loss_value


@tf.function
def computeGradient(xis, xjs, model: Model, temperature=0.1):
    with tf.GradientTape() as tape:
        loss_value = contrastiveLoss(xis, xjs, model, temperature)
    return loss_value, tape.gradient(loss_value, model.trainable_variables)


# Data augmentation
datagen = ImageDataGenerator(fill_mode="constant",
                             width_shift_range=0.3,
                             height_shift_range=0.3,
                             zoom_range=[0.4, 0.9],
                             channel_shift_range=0.5)

g1 = datagen.flow(x_train, batch_size=BATCH_SIZE, shuffle=False)
g2 = datagen.flow(x_train, batch_size=BATCH_SIZE, shuffle=False)

# d1 = g1.next()
# d2 = g2.next()
# max_img_num = 10
# imgs = []
# for i in range(max_img_num):
#     imgs.append(image.array_to_img(x_train[i], scale=True))
#     imgs.append(image.array_to_img(d1[i], scale=True))
#     imgs.append(image.array_to_img(d2[i], scale=True))
#
# show_imgs(imgs, row=10, col=3)

for epoch in range(EPOCHS_PRIOR):
    epoch_loss_avg = Mean()

    for i, (x1, x2) in enumerate(zip(g1, g2)):
        loss_value, grad = computeGradient(x1, x2, CL_model, temperature=T)
        OPTIMIZER.apply_gradients(zip(grad, CL_model.trainable_variables))
        epoch_loss_avg(loss_value)
        # print(i)
        if i > (x_train.shape[0] // BATCH_SIZE):
            break

    print("Epoch {}/{} Loss: {:.3f}".format(epoch + 1, EPOCHS_PRIOR, epoch_loss_avg.result().numpy()))
