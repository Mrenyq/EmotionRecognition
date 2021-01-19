import tensorflow as tf
from ContrastiveLearning.Models.ECGandEEGEncoder import ECGEEGEncoder
from Conf.Settings import FEATURES_N, DATASET_PATH, CHECK_POINT_PATH, TENSORBOARD_PATH, ECG_RAW_N, EEG_RAW_N, EEG_RAW_CH
from KnowledgeDistillation.Utils.DataFeaturesGenerator import DataFetchPreTrain_CL
import os

# setting
num_output = 4
initial_learning_rate = 0.55e-3
EPOCHS = 500
PRE_EPOCHS = 100
BATCH_SIZE = 128
T = 0.1
th = 0.5
wait = 10
EXPECTED_ECG_SIZE = (96, 96)

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

    training_data = DATASET_PATH + "training_data_" + str(fold) + ".csv"
    validation_data = DATASET_PATH + "validation_data_" + str(fold) + ".csv"
    testing_data = DATASET_PATH + "test_data_" + str(fold) + ".csv"

    data_fetch = DataFetchPreTrain_CL(training_data, validation_data, testing_data, ECG_RAW_N)
    generator = data_fetch.fetch

    train_generator = tf.data.Dataset.from_generator(
        lambda: generator(training_mode=0),
        output_types=(tf.float32, tf.float32),
        output_shapes=(tf.TensorShape([ECG_RAW_N]), tf.TensorShape([EEG_RAW_N, EEG_RAW_CH])))

    val_generator = tf.data.Dataset.from_generator(
        lambda: generator(training_mode=1),
        output_types=(tf.float32, tf.float32),
        output_shapes=(tf.TensorShape([ECG_RAW_N]), tf.TensorShape([EEG_RAW_N, EEG_RAW_CH])))

    test_generator = tf.data.Dataset.from_generator(
        lambda: generator(training_mode=2),
        output_types=(tf.float32, tf.float32),
        output_shapes=(tf.TensorShape([ECG_RAW_N]), tf.TensorShape([EEG_RAW_N, EEG_RAW_CH])))

    # train dataset
    train_data = train_generator.shuffle(data_fetch.train_n).repeat(3).padded_batch(
        BATCH_SIZE, padded_shapes=(tf.TensorShape([ECG_RAW_N]), tf.TensorShape([EEG_RAW_N, EEG_RAW_CH])))

    val_data = val_generator.padded_batch(
        BATCH_SIZE, padded_shapes=(tf.TensorShape([ECG_RAW_N]), tf.TensorShape([EEG_RAW_N, EEG_RAW_CH])))

    test_data = test_generator.padded_batch(
        BATCH_SIZE, padded_shapes=(tf.TensorShape([ECG_RAW_N]), tf.TensorShape([EEG_RAW_N, EEG_RAW_CH])))

    CL = ECGEEGEncoder()
    input_ecg = tf.keras.layers.Input(shape=(ECG_RAW_N,))
    input_eeg = tf.keras.layers.Input(shape=(EEG_RAW_N, EEG_RAW_CH))
    ecg_model, eeg_model, ecg_encoder, eeg_encoder = CL.createModel(input_ecg, input_eeg)
    eeg_model.summary()
    learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=initial_learning_rate,
                                                                   decay_steps=EPOCHS, decay_rate=0.95,
                                                                   staircase=True)
    # optimizer = tf.keras.optimizers.SGD(learning_rate=initial_learning_rate)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    # ---------------------------Epoch&Loss--------------------------#
    # loss
    train_loss = tf.keras.metrics.Mean()
    vald_loss = tf.keras.metrics.Mean()

    # pre_trained_loss = tf.keras.metrics.Mean()

    # Manager
    # checkpoint = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, base_model=model)
    # manager = tf.train.CheckpointManager(checkpoint, checkpoint_prefix, max_to_keep=3)
    # checkpoint.restore(manager.latest_checkpoint)

    def train_step(inputs, GLOBAL_BATCH_SIZE=0, temperature=0.1):
        x_ecg = inputs[0]
        x_eeg = inputs[1]

        with tf.GradientTape() as tape:
            final_loss = CL.computeAvgLoss(x_ecg, x_eeg, global_batch_size=GLOBAL_BATCH_SIZE, temperature=temperature)
        # update gradient
        update_weights = CL.ecg_model.trainable_variables + CL.eeg_model.trainable_variables
        grads = tape.gradient(final_loss, update_weights)
        optimizer.apply_gradients(zip(grads, update_weights))

        train_loss(final_loss)

        return final_loss

    def test_step(inputs, GLOBAL_BATCH_SIZE=0, temperature=0.1):
        x_ecg = inputs[0]
        x_eeg = inputs[1]
        final_loss = CL.computeAvgLoss(x_ecg, x_eeg, global_batch_size=GLOBAL_BATCH_SIZE, temperature=temperature)
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


    it = 0
    for epoch in range(EPOCHS):
        # TRAIN LOOP
        total_loss = 0.0
        num_batches = 0
        for step, train in enumerate(train_data):
            # print(tf.reduce_max(train[0][0]))
            train_step(train, BATCH_SIZE, temperature=T)
            it += 1

        # with train_summary_writer.as_default():
        #     tf.summary.scalar('Loss', train_loss.result(), step=epoch)
        #     tf.summary.scalar('Arousal accuracy', train_ar_acc.result(), step=epoch)
        #     tf.summary.scalar('Valence accuracy', train_val_acc.result(), step=epoch)

        for step, val in enumerate(val_data):
            test_step(val, data_fetch.val_n, temperature=T)

        # with test_summary_writer.as_default():
        #     tf.summary.scalar('Loss', vald_loss.result(), step=epoch)
        #     tf.summary.scalar('Arousal accuracy', vald_ar_acc.result(), step=epoch)
        #     tf.summary.scalar('Valence accuracy', vald_val_acc.result(), step=epoch)

        template = (
            "epoch {} | Train_loss: {} | Val_loss: {}")
        print(template.format(epoch + 1, train_loss.result().numpy(), vald_loss.result().numpy()))

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
    for step, test in enumerate(test_data):
        test_step(test, data_fetch.test_n)
    template = "Test: loss: {}"
    print(template.format(vald_loss.result().numpy()))

    vald_reset_states()
    print("-----------------------------------------------------------------------------------------")
