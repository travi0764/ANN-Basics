import tensorflow_datasets as tfds
import logging

def get_data(dataset,validation_datasize):
    
    (X_train_full, y_train_full), (X_test, y_test) = tfds.as_numpy(tfds.load(dataset, split = ['train', 'test'], 
    batch_size=-1, as_supervised=True))
    logging.info("Dataset downloaded and Loaded in Variables")
    logging.info("--"*30)
    X_valid, X_train = X_train_full[:validation_datasize] / 255., X_train_full[validation_datasize:] / 255.
    y_valid, y_train = y_train_full[:validation_datasize], y_train_full[validation_datasize:]

    X_test = X_test / 255.
    logging.info("Data set points scaled to 0-1")
    return (X_train, y_train), (X_valid, y_valid), (X_test, y_test)