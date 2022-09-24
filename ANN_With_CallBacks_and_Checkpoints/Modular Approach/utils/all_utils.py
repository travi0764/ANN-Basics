import time
import os
import pandas as pd
import matplotlib.pyplot as plt
import yaml
import logging

def get_unique_filename(name):
    timestamp = time.asctime().replace(" ", "_").replace(":", "_")
    unique_name = f"{timestamp}_at_{name}"

    return unique_name


def save_model(model, model_name, model_dir):
    unique_file_name = get_unique_filename(model_name)
    path_to_dir = os.path.join(model_dir,unique_file_name)
    logging.info(f"Saving the model at {path_to_dir}")
    model.save(path_to_dir)

def save_plot(history,plot_dir,dataset):
    plot_name = timestamp = time.asctime().replace(" ", "_").replace(":", "_")
    plotPath = os.path.join(plot_dir, plot_name) # model/filename
    df = pd.DataFrame(history.history)
    df.plot(figsize = (10,7))

    plt.grid(True)
    plt.title(f"Dataset : {dataset}")
    plt.savefig(plotPath)
    logging.info(f"Saving the plot at {plotPath}")


def read_config(config_path):
    with open(config_path) as config_file:
        content = yaml.safe_load(config_file)

    return content

# def initialize_tensorboard(config, X_train, input_size, action = "train"):

#     logs = config['logs']
#     unique_dir_name = get_unique_filename("tb_logs")
#     Tensorboard_root_log_dir = os.path.join(logs["log_dir"],logs['tensorboard_logs'],action,unique_dir_name)
#     os.makedirs(Tensorboard_root_log_dir,exist_ok=True)

#     file_writer = tf.summary.create_file_writer(logdir = Tensorboard_root_log_dir)
#     input_size.insert(0,-1)
#     input_size = tuple(input_size)

#     logging.info("20 Train Samples Plotted on Tensorboard")
#     with file_writer.as_default():
#         images = np.reshape(X_train[10:30], input_size)
#         tf.summary.image("20 Samples Dataset",images, max_outputs=25, step = 0)

#     return Tensorboard_root_log_dir