import tensorflow as tf
import logging

def create_model(LOSS_FUNCTION,METRICS,OPTIMIZER,NUM_CLASSES,INPUT_SIZE):

    LAYERS = [
        tf.keras.layers.Flatten(input_shape = INPUT_SIZE, name  = "inputLAYER"),
        tf.keras.layers.Dense(512, activation  = "relu", name  = "hiddenLAYER1"),
        tf.keras.layers.Dense(256, activation  = "relu", name  = "hiddenLAYER2"),
        tf.keras.layers.Dense(128, activation  = "relu", name  = "hiddenLAYER3"),
        tf.keras.layers.Dense(64, activation  = "relu", name  = "hiddenLAYER4"),
        tf.keras.layers.Dense(NUM_CLASSES, activation  = "softmax", name  = "outputLAYER"),
    ]

    model_clf = tf.keras.models.Sequential(LAYERS)

    model_clf.summary()
    
    logging.info(f"Model Summary : {model_clf.summary()}")
    model_clf.compile(loss = LOSS_FUNCTION, optimizer = OPTIMIZER, metrics = METRICS)
    logging.info("Model Compiled Successfully.")

    return model_clf  ## Untrained model, but compiled 


