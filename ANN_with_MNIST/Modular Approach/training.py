import os
from utils.all_utils import save_model,read_config,save_plot
from utils.data_management import get_data
from utils.model_initialization import create_model
import argparse
import logging

def training(config_path):
    logging_str = "[%(asctime)s: %(levelname)s: %(module)s ] %(message)s"
    config = read_config(config_path)
    
    logs_dir = config['logs']['log_dir']
    general_logs = config['logs']['general_logs']
    general_logs_dir = os.path.join(logs_dir,general_logs)

    os.makedirs(general_logs_dir, exist_ok=True)
    logging.basicConfig(filename=os.path.join(general_logs_dir,"running_logs.log"),level=logging.INFO,format = logging_str,filemode="a")

    dataset = config['params']['dataset']
    logging.info(f"Training Dataset Name : {dataset}")
    validation_datasize = config['params']['validation_datasize']
    logging.info(f"Validation Data Size : {validation_datasize}")
    (X_train, y_train), (X_valid, y_valid), (X_test, y_test) = get_data(dataset, validation_datasize)
    
    logging.info(f"DataSet is splitted and stoed in X_Train, X_valid, X_test")
    logging.info(f"X_train Size : {X_train.shape}")
    logging.info(f"X_Valid Size : {X_valid.shape}")
    logging.info(f"X_test Size : {X_test.shape}")
    
    LOSS_FUNCTION = config['params']['loss_function']
    OPTIMIZER = config['params']['optimizer']
    METRICS = config['params']['metrics']
    NUM_CLASSES = config['params']['num_classes']
    INPUT_SIZE = config['params']['input_size'].split(",")
    EPOCHS = config['params']['epochs']
    VALIDATION_SET = (X_valid, y_valid)
    INPUT_SIZE = [int(i) for i in INPUT_SIZE]
    
    logging.info(f"Model Parameters : ")
    logging.info(f"Loss Function : {LOSS_FUNCTION} \n OPTIMIZER : {OPTIMIZER} \n METRICS : {METRICS} \n INPUT_SIZE : {INPUT_SIZE} \n EPOCHS : {EPOCHS} \n NUM_CLASSES : {NUM_CLASSES} ")
    model = create_model(LOSS_FUNCTION,METRICS,OPTIMIZER,NUM_CLASSES,INPUT_SIZE)
    history = model.fit(X_train, y_train, epochs = EPOCHS, validation_data= VALIDATION_SET)

    logging.info(f"Model History : \n {history}")
    artifacts_dir = config['artifacts']['artifacts_dir']
    model_dir = config['artifacts']['model_dir']
    model_name = config['artifacts']['model_name']
    plot_dir = config['artifacts']['plot_dir']

    model_dir_path = os.path.join(artifacts_dir, model_dir)
    os.makedirs(model_dir_path, exist_ok=True)

    plot_dir_path = os.path.join(artifacts_dir, plot_dir)
    os.makedirs(plot_dir_path, exist_ok=True)

    save_plot(history, plot_dir_path,dataset)

    save_model(model,model_name,model_dir_path)


if __name__ == "__main__" :
    args = argparse.ArgumentParser()

    args.add_argument("--config","-c",default= "config.yaml")
    parsed_args = args.parse_args()
    
    try:
        print(">>>>> Starting Training >>>>>")
        training(config_path=parsed_args.config)
        print("<<<<< Training Done Successfully <<<<<<")
    except Exception as e:
        logging.exception(e)
        raise e

