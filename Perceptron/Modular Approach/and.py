from utils.model import Perceptron
from utils.all_utills import prepare_data, save_model,save_plot
import pandas as pd
import os
import logging

logging_str = "[%(asctime)s: %(levelname)s: %(module)s ] %(message)s"
log_dir = "logs"

os.makedirs(log_dir,exist_ok=True)
logging.basicConfig(filename=os.path.join(log_dir,"running_logs.log"),level=logging.INFO,format = logging_str,filemode="a")


def main(data:dict, modelName:str, plotName:str, eta:float, epochs:int) -> None :
    """main function to start training and saving models and plots

    Args:
        data (dict): data to be passed to model
        modelName (str): model name to be saved as 
        plotName (str): plot name to be saved as
        eta (float): learning rate for the model
        epochs (int): number of epochs to be trained 
    """
    
    df = pd.DataFrame(data)
    logging.info(f"This is actual dataframe : {df}")
    X,y = prepare_data(df)
    model = Perceptron(eta = eta, epochs=epochs)
    model.fit(X,y)
    save_model(model,filename=modelName)
    save_plot(df,plotName,model)


if __name__== "__main__":

    AND = {
        "x1": [0,0,1,1],
        "x2": [0,1,0,1],
        "y": [0,0,0,1],
    }



    ETA = 0.3 # 0 and 1
    EPOCHS = 10

    try:
        print(">>>>> Starting Training >>>>>")
        main(data = AND,modelName="and.model",plotName="and.png",eta = ETA,epochs=EPOCHS)
        print("<<<<< Training Done Successfully <<<<<<")
    except Exception as e:
        logging.exception(e)
        raise e
