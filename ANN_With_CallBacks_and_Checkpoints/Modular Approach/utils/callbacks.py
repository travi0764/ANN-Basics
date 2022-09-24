import tensorflow as tf
import os
import numpy as np
import logging
from utils.all_utils import get_unique_filename

def get_callbacks(config, X_train):
    INPUT_SIZE = config['params']['input_size'].split(",")
    INPUT_SIZE = [int(i) for i in INPUT_SIZE]
    datasets = config['params']['dataset']

    logs = config['logs']
    unique_dir_name = get_unique_filename("tb_logs")
    TENSORBOARD_LOG_DIR = os.path.join(logs["log_dir"],logs['tensorboard_logs'],datasets, unique_dir_name)
    os.makedirs(TENSORBOARD_LOG_DIR,exist_ok=True)

    file_writer = tf.summary.create_file_writer(logdir = TENSORBOARD_LOG_DIR)
    INPUT_SIZE.insert(0,-1)
    INPUT_SIZE = tuple(INPUT_SIZE)

    logging.info("20 Train Samples Plotted on Tensorboard")
    with file_writer.as_default():
        images = np.reshape(X_train[10:30], INPUT_SIZE)
        tf.summary.image("20 Samples Dataset",images, max_outputs=25, step = 0)

    tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir = TENSORBOARD_LOG_DIR)

    params = config['params']

    early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience= params['patience'], restore_best_weights= params['restore_best_weights'])

    artifacts = config['artifacts']

    CKPT_DIR = os.path.join(artifacts['artifacts_dir'],artifacts['checkpoint_dir'])
    os.makedirs(CKPT_DIR,exist_ok= True)

    checkpointing_cb = tf.keras.callbacks.ModelCheckpoint(CKPT_DIR, "ckpt_model.h5")

    return [tensorboard_cb, early_stopping_cb, checkpointing_cb]