import tensorflow as tf
import matplotlib.pyplot as plt
from ContrastiveLearning.Models.ECGandEEGEncoder import ECGEEGEncoder
from Conf.Settings import FEATURES_N, DATASET_PATH, CHECK_POINT_PATH, TENSORBOARD_PATH, ECG_RAW_N, EEG_RAW_N, EEG_RAW_CH
from KnowledgeDistillation.Utils.DataFeaturesGenerator import DataFetchPreTrain_CL
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus)
if gpus:
    # Create 4 virtual GPU
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPU,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Virtual devices must be set before GPUs have been initialized
        print(e)

cross_tower_ops = tf.distribute.HierarchicalCopyAllReduce(num_packs=2)
strategy = tf.distribute.MirroredStrategy(cross_device_ops=cross_tower_ops)

# setting
num_output = 4
initial_learning_rate = 0.001
EPOCHS = 500
PRE_EPOCHS = 100
BATCH_SIZE = 200
T = 0.1
ALL_BATCH_SIZE = BATCH_SIZE * strategy.num_replicas_in_sync

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
    train_data = train_generator.shuffle(data_fetch.train_n).padded_batch(
        BATCH_SIZE, padded_shapes=(tf.TensorShape([ECG_RAW_N]), tf.TensorShape([EEG_RAW_N, EEG_RAW_CH])))

    val_data = val_generator.padded_batch(
        BATCH_SIZE, padded_shapes=(tf.TensorShape([ECG_RAW_N]), tf.TensorShape([EEG_RAW_N, EEG_RAW_CH])))

    test_data = test_generator.padded_batch(
        BATCH_SIZE, padded_shapes=(tf.TensorShape([ECG_RAW_N]), tf.TensorShape([EEG_RAW_N, EEG_RAW_CH])))

    with strategy.scope():
        # model = EnsembleStudent(num_output=num_output, expected_size=EXPECTED_ECG_SIZE)

        # load pretrained model
        # checkpoint_prefix_base = CHECK_POINT_PATH + "fold_M" + str(fold)

        CL = ECGEEGEncoder()
        input_ecg = tf.keras.layers.Input(shape=(ECG_RAW_N,))
        input_eeg = tf.keras.layers.Input(shape=(EEG_RAW_N, EEG_RAW_CH))
        ecg_model, eeg_model, ecg_encoder, eeg_encoder = CL.createModel(input_ecg, input_eeg)
        eeg_model.summary()
        # learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=initial_learning_rate,
        #                                                                decay_steps=EPOCHS, decay_rate=0.95,
        #                                                                staircase=True)
        learning_rate = initial_learning_rate
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        # ---------------------------Epoch&Loss--------------------------#
        # loss
        train_loss = tf.keras.metrics.Mean()
        vald_loss = tf.keras.metrics.Mean()

    # Manager
    # checkpoint = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, base_model=model)
    # manager = tf.train.CheckpointManager(checkpoint, checkpoint_prefix, max_to_keep=3)
    # checkpoint.restore(manager.latest_checkpoint)

    with strategy.scope():
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

    with strategy.scope():
        # `experimental_run_v2` replicates the provided computation and runs it
        # with the distributed input.

        @tf.function
        def distributed_train_step(dataset_inputs, GLOBAL_BATCH_SIZE, temperature=0.1):
            per_replica_losses = strategy.run(train_step,
                                              args=(dataset_inputs, GLOBAL_BATCH_SIZE, temperature))
            return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,
                                   axis=None)

        @tf.function
        def distributed_test_step(dataset_inputs, GLOBAL_BATCH_SIZE, temperature=0.1):
            per_replica_losses = strategy.run(test_step,
                                              args=(dataset_inputs, GLOBAL_BATCH_SIZE, temperature))
            return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,
                                   axis=None)


        it = 0
        train_loss_history = []
        val_loss_history = []

        for epoch in range(EPOCHS):
            # TRAIN LOOP
            total_loss = 0.0
            num_batches = 0
            for step, train in enumerate(train_data):
                # print(tf.reduce_max(train[0][0]))
                distributed_train_step(train, ALL_BATCH_SIZE, temperature=T)
                it += 1

            # with train_summary_writer.as_default():
            #     tf.summary.scalar('Loss', train_loss.result(), step=epoch)
            #     tf.summary.scalar('Arousal accuracy', train_ar_acc.result(), step=epoch)
            #     tf.summary.scalar('Valence accuracy', train_val_acc.result(), step=epoch)

            for step, val in enumerate(val_data):
                distributed_test_step(val, ALL_BATCH_SIZE, temperature=T)

            # with test_summary_writer.as_default():
            #     tf.summary.scalar('Loss', vald_loss.result(), step=epoch)
            #     tf.summary.scalar('Arousal accuracy', vald_ar_acc.result(), step=epoch)
            #     tf.summary.scalar('Valence accuracy', vald_val_acc.result(), step=epoch)

            train_loss_history.append(train_loss.result().numpy())
            val_loss_history.append(vald_loss.result().numpy())

            template = "epoch {}/{} | Train_loss: {} | Val_loss: {}"
            print(template.format(epoch + 1, EPOCHS, train_loss.result().numpy(), vald_loss.result().numpy()))

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
        distributed_test_step(test, ALL_BATCH_SIZE)
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
