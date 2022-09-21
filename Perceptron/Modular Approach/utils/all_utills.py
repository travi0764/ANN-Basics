import logging
import pandas as pd
import numpy as np
import joblib
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import os

plt.style.use("fivethirtyeight")
def save_model(model, filename):
    """saving the model in current directory

    Args:
        model (model.Perceptron): trained model of class Perceptron
        filename (str): model to be saved as filename
    """
    logging.info("Saving the trained Model")
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True) # ONLY CREATE IF MODEL_DIR DOESN"T EXISTS
    filePath = os.path.join(model_dir, filename) # model/filename
    joblib.dump(model, filePath)
    logging.info(f"Saving the trained Model at {filePath}")

def prepare_data(df:pd.DataFrame) -> tuple(pd.DataFrame, pd.Series):
    """seperates dependent and independent variables from data dataframe

    Args:
        df (pd.DataFrame): dataframe which containes dependent and independent variables

    Returns:
        pd.DataFrame: dataframe containing all dependent features
        pd.Series: series containing all independent features

    """
    logging.info("Preparing data by segregationg dependent and independent variables")
    X = df.drop("y",axis=1)
    y = df['y']
    return X,y

def save_plot(df:pd.DataFrame, file_name:str, model) -> None:
    """ saves decision boundary of trained models in current directory

    Args:
        df (pd.DataFrame): dataframe which containes dependent and independent variables
        file_name (str): img to be saved as file_name

    """
    def _create_base_plot(df:pd.DataFrame) -> None:
        """creates base plot for trained model decision boundary 

        Args:
            df (pd.DataFrame): dataframe which containes dependent and independent variables
        """
        logging.info("Creating the base plot")
        df.plot(kind="scatter", x="x1", y="x2", c="y", s=100, cmap="winter")
        plt.axhline(y=0, color="black", linestyle="--", linewidth=1)
        plt.axvline(x=0, color="black", linestyle="--", linewidth=1)
        figure = plt.gcf() # get current figure
        figure.set_size_inches(10, 8)

    def _plot_decision_regions(X, y, classfier, resolution=0.02):
        """plots decision boundary of trained model on base plot

        Args:
            X (pd.DataFrame): dataframe of dependent features of data
            y (pd.Series): series of independent feature of data
            classfier (_type_): trained model
            resolution (float, optional): resolution of plot. Defaults to 0.02.
        """
        logging.info("Plotting the decision region")
        colors = ("red", "blue", "lightgreen", "gray", "cyan")
        cmap = ListedColormap(colors[: len(np.unique(y))])

        X = X.values # as a array
        x1 = X[:, 0] 
        x2 = X[:, 1]
        x1_min, x1_max = x1.min() -1 , x1.max() + 1
        x2_min, x2_max = x2.min() -1 , x2.max() + 1  

        xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), 
                            np.arange(x2_min, x2_max, resolution))
        Z = classfier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
        Z = Z.reshape(xx1.shape)
        plt.contourf(xx1, xx2, Z, alpha=0.2, cmap=cmap)
        plt.xlim(xx1.min(), xx1.max())
        plt.ylim(xx2.min(), xx2.max())

        plt.plot()
    X, y = prepare_data(df)

    _create_base_plot(df)
    _plot_decision_regions(X, y, model)

    plot_dir = "plots"
    os.makedirs(plot_dir, exist_ok=True) # ONLY CREATE IF MODEL_DIR DOESN"T EXISTS
    plotPath = os.path.join(plot_dir, file_name) # model/filename
    plt.savefig(plotPath)
    logging.info(f"Saving the lot at {plotPath}")