import tensorflow as tf


# Functional API
class ECGEEGEncoder:
    def __init__(self, dim_head_output=128):
        self.dim_head_output = dim_head_output
        self.ecg_model = None
        self.eeg_model = None
        self.ecg_encoder = None
        self.eeg_encoder = None

    def ecgEncoder(self, input_tensor, pretrain=True):
        x = tf.expand_dims(input_tensor, axis=-1)

        # Encoder
        for f in [4, 8, 16, 32, 32]:
            x = tf.keras.layers.Conv1D(filters=f, kernel_size=5, strides=1, padding="same", trainable=pretrain)(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.ELU()(x)
            x = tf.keras.layers.MaxPooling1D(pool_size=4)(x)
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        h_ecg = x

        # Head
        for u in [128, 128]:
            x = tf.keras.layers.Dense(units=u)(x)
            x = tf.keras.layers.ELU()(x)
            x = tf.keras.layers.Dropout(0.15)(x)
        z_ecg = tf.keras.layers.Dense(units=self.dim_head_output)(x)

        return h_ecg, z_ecg

    def eegEncoder3D(self, input_tensor, pretrain=True):

        # Encoder
        x = tf.keras.layers.Conv3D(filters=64, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding="same",
                                   trainable=pretrain)(input_tensor)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ELU()(x)
        x = tf.keras.layers.MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2))(x)
        x = tf.keras.layers.Conv3D(filters=128, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding="same",
                                   trainable=pretrain)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ELU()(x)
        x = tf.keras.layers.MaxPooling3D(pool_size=(2, 2, 2), strides=(1, 2, 2))(x)
        x = tf.keras.layers.Conv3D(filters=256, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding="same",
                                   trainable=pretrain)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ELU()(x)
        x = tf.keras.layers.Conv3D(filters=256, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding="same",
                                   trainable=pretrain)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ELU()(x)
        x = tf.keras.layers.MaxPooling3D(pool_size=(2, 2, 2), strides=(1, 2, 2))(x)
        h_eeg = tf.keras.layers.GlobalAveragePooling3D()(x)

        # Head
        x = h_eeg
        for u in [1024, 512]:
            x = tf.keras.layers.Dense(units=u)(x)
            x = tf.keras.layers.ELU()(x)
            x = tf.keras.layers.Dropout(0.15)(x)
        z_eeg = tf.keras.layers.Dense(units=self.dim_head_output)(x)

        return h_eeg, z_eeg

    def eegEncoder1D(self, input_tensor, pretrain=True):

        # Encoder
        x = input_tensor
        for f in [16, 32, 64, 128, 128]:
            x = tf.keras.layers.Conv1D(filters=f, kernel_size=5, strides=1, padding="same", trainable=pretrain)(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.ELU()(x)
            x = tf.keras.layers.MaxPooling1D(pool_size=4)(x)
        h_eeg = tf.keras.layers.GlobalAveragePooling1D()(x)

        # Head
        x = h_eeg
        for u in [256, 256]:
            x = tf.keras.layers.Dense(units=u)(x)
            x = tf.keras.layers.ELU()(x)
            x = tf.keras.layers.Dropout(0.15)(x)
        z_eeg = tf.keras.layers.Dense(units=self.dim_head_output)(x)

        return h_eeg, z_eeg

    def createModel(self, input_ecg, input_eeg, pretrain=True):
        h_ecg, z_ecg = self.ecgEncoder(input_ecg, pretrain=pretrain)
        # h_eeg, z_eeg = self.eegEncoder3D(input_eeg, pretrain=pretrain)
        h_eeg, z_eeg = self.eegEncoder1D(input_eeg, pretrain=pretrain)
        self.ecg_model = tf.keras.models.Model(input_ecg, z_ecg)
        self.eeg_model = tf.keras.models.Model(input_eeg, z_eeg)
        self.ecg_encoder = tf.keras.models.Model(self.ecg_model.input, h_ecg)
        self.eeg_encoder = tf.keras.models.Model(self.eeg_model.input, h_eeg)
        return self.ecg_model, self.eeg_model, self.ecg_encoder, self.eeg_encoder

    def contrastiveLoss(self, input_ecg, input_eeg, temperature=0.1):
        x_ecg = tf.convert_to_tensor(input_ecg, dtype=tf.float32)
        x_eeg = tf.convert_to_tensor(input_eeg, dtype=tf.float32)
        temperature = tf.convert_to_tensor(temperature, dtype=tf.float32)
        z_ecg = self.ecg_model(x_ecg)
        z_eeg = self.eeg_model(x_eeg)
        z_ecg = tf.math.l2_normalize(z_ecg, axis=1)
        z_eeg = tf.math.l2_normalize(z_eeg, axis=1)
        z_all = tf.concat([z_ecg, z_eeg], axis=0)
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

        losses = []
        for k in range(batch_size_2 // 2):
            losses.append((loss[k, k + (batch_size_2 // 2)] + loss[k + (batch_size_2 // 2), k]) / 2)

        return losses

    def computeAvgLoss(self, input_ecg, input_eeg, global_batch_size, temperature=0.1):
        loss_value = self.contrastiveLoss(input_ecg, input_eeg, temperature)
        final_loss = tf.nn.compute_average_loss(loss_value, global_batch_size=global_batch_size)
        return final_loss


# Subclassing API
class Encoder(tf.keras.Model):

    def __init__(self, num_output=4, pretrain=True):
        super(Encoder, self).__init__(self)

        # ECG encoder
        self.ecg_conv1 = tf.keras.layers.Conv1D(filters=8, kernel_size=5, strides=1, activation=None, name="ecg_conv1",
                                                padding="same", trainable=pretrain)
        self.ecg_conv2 = tf.keras.layers.Conv1D(filters=8, kernel_size=5, strides=1, activation=None, name="ecg_conv2",
                                                padding="same", trainable=pretrain)
        self.ecg_conv3 = tf.keras.layers.Conv1D(filters=16, kernel_size=5, strides=1, activation=None, name="ecg_conv3",
                                                padding="same", trainable=pretrain)
        self.ecg_conv4 = tf.keras.layers.Conv1D(filters=16, kernel_size=5, strides=1, activation=None, name="ecg_conv4",
                                                padding="same", trainable=pretrain)
        self.ecg_conv5 = tf.keras.layers.Conv1D(filters=32, kernel_size=5, strides=1, activation=None, name="ecg_conv5",
                                                padding="same", trainable=pretrain)
        self.ecg_conv6 = tf.keras.layers.Conv1D(filters=32, kernel_size=5, strides=1, activation=None, name="ecg_conv6",
                                                padding="same", trainable=pretrain)

        # Batch norm
        self.ecg_batch_1 = tf.keras.layers.BatchNormalization(name="ecg_batch1")
        self.ecg_batch_2 = tf.keras.layers.BatchNormalization(name="ecg_batch2")
        self.ecg_batch_3 = tf.keras.layers.BatchNormalization(name="ecg_batch3")
        self.ecg_batch_4 = tf.keras.layers.BatchNormalization(name="ecg_batch4")
        self.ecg_batch_5 = tf.keras.layers.BatchNormalization(name="ecg_batch5")
        self.ecg_batch_6 = tf.keras.layers.BatchNormalization(name="ecg_batch6")

        # activation
        self.elu = tf.keras.layers.ELU()

        # head
        self.ecg_dense1 = tf.keras.layers.Dense(units=32, name="ecg_dense1")
        self.ecg_dense2 = tf.keras.layers.Dense(units=32, name="ecg_dense2")
        self.ecg_dense3 = tf.keras.layers.Dense(units=32, name="ecg_dense3")

        # flattent
        self.flat = tf.keras.layers.Flatten()

        # pool
        self.max_pool = tf.keras.layers.MaxPool1D(pool_size=3)

        # dropout
        self.ecg_dropout1 = tf.keras.layers.Dropout(0.15)
        self.ecg_dropout2 = tf.keras.layers.Dropout(0.15)
        self.ecg_dropout3 = tf.keras.layers.Dropout(0.15)

        # loss

        self.multi_cross_loss = tf.losses.CategoricalCrossentropy(from_logits=True,
                                                                  reduction=tf.keras.losses.Reduction.NONE)
        self.mean_square_loss = tf.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)

        # weight
        self.weight = tf.constant([0.8, 0.6, 0.8])

    def forward(self, x, dense, norm=None, activation=None):
        if norm is None:
            return activation(dense(x))
        return activation(norm(dense(x)))

    def call(self, inputs, training=None, mask=None):
        x = tf.expand_dims(inputs, -1)

        # encoder
        x = self.max_pool(self.forward(x, self.ecg_conv1, self.ecg_batch_1, self.elu))
        x = self.max_pool(self.forward(x, self.ecg_conv2, self.ecg_batch_2, self.elu))
        x = self.max_pool(self.forward(x, self.ecg_conv3, self.ecg_batch_3, self.elu))
        x = self.max_pool(self.forward(x, self.ecg_conv4, self.ecg_batch_4, self.elu))
        x = self.max_pool(self.forward(x, self.ecg_conv5, self.ecg_batch_5, self.elu))
        x = self.max_pool(self.forward(x, self.ecg_conv6, self.ecg_batch_6, self.elu))
        h_ecg = self.flat(x)

        # Head
        x = self.ecg_dropout1(self.elu(self.ecg_dense1(h_ecg)))
        x = self.ecg_dropout2(self.elu(self.ecg_dense2(x)))
        z_ecg = self.ecg_dropout3(self.elu(self.ecg_dense3(x)))

        return z_ecg, h_ecg

    def trainM(self, X, y_ar, y_val, y_ar_t, y_val_t, z_t, T, alpha, global_batch_size, training=False):
        z_ar, z_val, z = self.call(X, training=training)
        y_ar_t = tf.nn.softmax(y_ar_t / T, -1)
        y_val_t = tf.nn.softmax(y_val_t / T, -1)
        beta = 1 - alpha
        final_loss_ar = tf.nn.compute_average_loss((alpha * self.multi_cross_loss(y_ar, z_ar)) + (
                    beta * self.multi_cross_loss(y_ar_t, z_ar / T, sample_weight=self.weight)),
                                                   global_batch_size=global_batch_size)
        final_loss_val = tf.nn.compute_average_loss(
            (alpha * self.multi_cross_loss(y_val, z_ar, sample_weight=self.weight)) + (
                    beta * self.multi_cross_loss(y_val_t, z_ar / T)), global_batch_size=global_batch_size)

        prediction_ar = tf.argmax(tf.nn.softmax(z_ar, -1), -1)
        prediction_val = tf.argmax(tf.nn.softmax(z_val, -1), -1)

        final_loss = (0.5 * (final_loss_ar + final_loss_val))
        return final_loss, prediction_ar, prediction_val

    def test(self, X, y_ar, y_val, global_batch_size, training=False):
        z_ar, z_val, z = self.call(X, training=training)

        final_loss_ar = tf.nn.compute_average_loss(self.sparse_cross_loss(y_ar, z_ar),
                                                   global_batch_size=global_batch_size)
        final_loss_val = tf.nn.compute_average_loss(self.sparse_cross_loss(y_val, z_val),
                                                    global_batch_size=global_batch_size)
        prediction_ar = tf.argmax(tf.nn.softmax(z_ar, -1), -1)
        prediction_val = tf.argmax(tf.nn.softmax(z_val, -1), -1)
        final_loss = (0.5 * (final_loss_ar + final_loss_val))
        return final_loss, prediction_ar, prediction_val
