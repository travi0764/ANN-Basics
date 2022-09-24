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
