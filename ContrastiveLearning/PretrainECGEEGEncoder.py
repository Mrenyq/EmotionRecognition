import tensorflow as tf
import matplotlib.pyplot as plt
from ContrastiveLearning.Models.ECGandEEGEncoder import ECGEEGEncoder
from Conf.Settings import FEATURES_N, DATASET_PATH, CHECK_POINT_PATH, TENSORBOARD_PATH, ECG_RAW_N, EEG_RAW_N, EEG_RAW_CH
from KnowledgeDistillation.Utils.DataFeaturesGenerator import DataFetchPreTrain_CL
import os

# setting
num_output = 4
initial_learning_rate = 0.2e-3
EPOCHS = 500
PRE_EPOCHS = 100
BATCH_SIZE = 128
T = 0.1

for fold in range(1, 2):
    prev_val_loss = 1000
    wait_i = 0
    # checkpoint_prefix = CHECK_POINT_PATH + "KD\\fold_M" + str(fold)
    # tensorboard
    # current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    # train_log_dir = TENSORBOARD_PATH + "KD\\" + current_time + '/train'
    # test_log_dir = TENSORBOARD_PATH + "KD\\" + current_time + '/test'
    # train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    # test_summary_writer = tf.summary.create_file_writer(test_log_dir)

    CL = ECGEEGEncoder()
    input_ecg = tf.keras.layers.Input(shape=(ECG_RAW_N,))
    input_eeg = tf.keras.layers.Input(shape=(EEG_RAW_N, EEG_RAW_CH))
    ecg_model, eeg_model, ecg_encoder, eeg_encoder = CL.createModel(input_ecg, input_eeg)
    ecg_model.summary()
    eeg_model.summary()

    training_data = DATASET_PATH + "training_data_" + str(fold) + ".csv"
    validation_data = DATASET_PATH + "validation_data_" + str(fold) + ".csv"
    testing_data = DATASET_PATH + "test_data_" + str(fold) + ".csv"

    # data_fetch = DataFetchPreTrain_CL(training_data, validation_data, testing_data, ECG_RAW_N)
    data_fetch = DataFetchPreTrain_CL(validation_data, validation_data, testing_data, ECG_RAW_N)
    generator = data_fetch.fetch

    train_generator_ecg = tf.data.Dataset.from_generator(
        lambda: generator(training_mode=0, ecg_or_eeg=0),
        output_types=(tf.float32, tf.int32),
        output_shapes=(tf.TensorShape([ECG_RAW_N]), tf.TensorShape([1])))

    val_generator_ecg = tf.data.Dataset.from_generator(
        lambda: generator(training_mode=1, ecg_or_eeg=0),
        output_types=(tf.float32, tf.int32),
        output_shapes=(tf.TensorShape([ECG_RAW_N]), tf.TensorShape([1])))

    test_generator_ecg = tf.data.Dataset.from_generator(
        lambda: generator(training_mode=2, ecg_or_eeg=0),
        output_types=(tf.float32, tf.int32),
        output_shapes=(tf.TensorShape([ECG_RAW_N]), tf.TensorShape([1])))

    train_generator_eeg = tf.data.Dataset.from_generator(
        lambda: generator(training_mode=0, ecg_or_eeg=1),
        output_types=(tf.float32, tf.int32),
        output_shapes=(tf.TensorShape([EEG_RAW_N, EEG_RAW_CH]), tf.TensorShape([1])))

    val_generator_eeg = tf.data.Dataset.from_generator(
        lambda: generator(training_mode=1, ecg_or_eeg=1),
        output_types=(tf.float32, tf.int32),
        output_shapes=(tf.TensorShape([EEG_RAW_N, EEG_RAW_CH]), tf.TensorShape([1])))

    test_generator_eeg = tf.data.Dataset.from_generator(
        lambda: generator(training_mode=2, ecg_or_eeg=1),
        output_types=(tf.float32, tf.int32),
        output_shapes=(tf.TensorShape([EEG_RAW_N, EEG_RAW_CH]), tf.TensorShape([1])))

    train_data_ecg = train_generator_ecg.shuffle(data_fetch.train_n).batch(BATCH_SIZE)
    val_data_ecg = val_generator_ecg.batch(BATCH_SIZE)
    test_data_ecg = test_generator_ecg.batch(BATCH_SIZE)
    train_data_eeg = train_generator_eeg.shuffle(data_fetch.train_n).batch(BATCH_SIZE)
    val_data_eeg = val_generator_eeg.batch(BATCH_SIZE)
    test_data_eeg = test_generator_eeg.batch(BATCH_SIZE)

    learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=initial_learning_rate,
                                                                   decay_steps=EPOCHS, decay_rate=0.95,
                                                                   staircase=True)
    # learning_rate = initial_learning_rate
    # optimizer = tf.keras.optimizers.SGD(learning_rate=initial_learning_rate)
    optimizer = tf.keras.optimizers.Adamax(learning_rate=learning_rate)

    # ---------------------------Epoch&Loss--------------------------#
    # loss
    train_loss = tf.keras.metrics.Mean()
    vald_loss = tf.keras.metrics.Mean()

    # pre_trained_loss = tf.keras.metrics.Mean()

    # Manager
    # checkpoint = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, base_model=model)
    # manager = tf.train.CheckpointManager(checkpoint, checkpoint_prefix, max_to_keep=3)
    # checkpoint.restore(manager.latest_checkpoint)

    @tf.function
    def train_step(x_ecg, x_eeg, label_ecg, label_eeg):

        with tf.GradientTape() as tape:
            final_loss = CL.contrastiveLoss(x_ecg, x_eeg, label_ecg, label_eeg)

        # update gradient
        grads = tape.gradient(final_loss, ecg_model.trainable_variables + eeg_model.trainable_variables)
        optimizer.apply_gradients(zip(grads, ecg_model.trainable_variables + eeg_model.trainable_variables))

        # update_weights = CL.ecg_model.trainable_variables + CL.eeg_model.trainable_variables
        # grads = tape.gradient(final_loss, update_weights)
        # optimizer.apply_gradients(zip(grads, update_weights))

        train_loss(final_loss)

        return final_loss

    @tf.function
    def test_step(x_ecg, x_eeg, label_ecg, label_eeg):
        final_loss = CL.contrastiveLoss(x_ecg, x_eeg, label_ecg, label_eeg)
        vald_loss(final_loss)

        return final_loss


    def train_reset_states():
        train_loss.reset_states()
        # train_ar_acc.reset_states()
        # train_val_acc.reset_states()


    def vald_reset_states():
        vald_loss.reset_states()
        # vald_ar_acc.reset_states()
        # vald_val_acc.reset_states()

    train_loss_history = []
    val_loss_history = []
    for epoch in range(EPOCHS):
        # TRAIN LOOP
        total_loss = 0.0
        num_batches = 0
        for step, train in enumerate(train_data_ecg):
            # print(tf.reduce_max(train[0][0]))
            train_step(train, BATCH_SIZE, temperature=T)

        # with train_summary_writer.as_default():
        #     tf.summary.scalar('Loss', train_loss.result(), step=epoch)
        #     tf.summary.scalar('Arousal accuracy', train_ar_acc.result(), step=epoch)
        #     tf.summary.scalar('Valence accuracy', train_val_acc.result(), step=epoch)

        for step, val in enumerate(val_data_ecg):
            test_step(val, BATCH_SIZE, temperature=T)

        # with test_summary_writer.as_default():
        #     tf.summary.scalar('Loss', vald_loss.result(), step=epoch)
        #     tf.summary.scalar('Arousal accuracy', vald_ar_acc.result(), step=epoch)
        #     tf.summary.scalar('Valence accuracy', vald_val_acc.result(), step=epoch)

        train_loss_history.append(train_loss.result().numpy())
        val_loss_history.append(vald_loss.result().numpy())

        template = "epoch {}/{} | Train_loss: {} | Val_loss: {}"
        print(template.format(epoch + 1, EPOCHS, train_loss.result().numpy(), vald_loss.result().numpy()))
        lr_now = optimizer._decayed_lr(tf.float32).numpy()
        print("Now learning_rate:", lr_now)

        # Save model

        # if (prev_val_loss > vald_loss.result().numpy()):
        #     prev_val_loss = vald_loss.result().numpy()
        #     wait_i = 0
        #     manager.save()
        # else:
        #     wait_i += 1
        # if (wait_i == wait):
        #     break

        # reset state
        train_reset_states()
        vald_reset_states()

    print("-------------------------------------------Testing----------------------------------------------")
    for step, test in enumerate(test_data_ecg):
        test_step(test, BATCH_SIZE)
    template = "Test: loss: {}"
    print(template.format(vald_loss.result().numpy()))

    vald_reset_states()
    print("-----------------------------------------------------------------------------------------")

    # Save weights
    os.makedirs("./Weights", exist_ok=True)
    ecg_encoder.save_weights("./weights/ECG_encoder_param_" + str(fold) + ".hdf5")
    eeg_encoder.save_weights("./weights/EEG_encoder_param_" + str(fold) + ".hdf5")

    plt.figure()
    plt.plot(train_loss_history)
    plt.plot(val_loss_history)
    plt.title('Contrastive Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'])
    plt.show()
