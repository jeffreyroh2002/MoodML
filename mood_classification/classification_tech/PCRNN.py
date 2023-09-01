import json
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.metrics import confusion_matrix

####EDIT BEFORE RUNNING ###########
NUM_CLASSES = 4

# path to json file that stores MFCCs and genre labels for each processed segment
DATA_PATH = "../snapmuse_7/snapmuse_7.json"
SAVE_MODEL = True
SAVE_HM = True

#OUTPUT DIR/FILE NAMES
NEWDIR_PATH = "../results/901_PCRNN_2D_snapmuse_7"

MODEL_NAME = "saved_model"
HM_NAME = "heatmap.png"
A_PLOT_NAME = 'accuracy.png'
L_PLOT_NAME = 'loss.png'

# Hyperparameters
LEARNING_RATE = 0.0001
EPOCHS = 30

####################################
tf.debugging.set_log_device_placement(True)

if not os.path.exists(NEWDIR_PATH):
    os.makedirs(NEWDIR_PATH)

def load_data(data_path):
    with open(data_path, "r") as fp:
        data = json.load(fp)

    # convert lists to numpy arrays
    X = np.array(data["mfcc"])
    y = np.array(data["labels"])
    label_list = data.get("mapping", {})   #Jazz, Classical, etc

    print(label_list)

    print("Data successfully loaded!")

    return X, y, label_list

def step_decay(epoch):
    initial_lr = LEARNING_RATE
    drop = 0.5
    epochs_drop = 10
    new_lr = initial_lr * np.power(drop, np.floor((1 + epoch) / epochs_drop))
    return new_lr

def save_plot(history, newdir_path=NEWDIR_PATH, a_plot_name=A_PLOT_NAME, l_plot_name=L_PLOT_NAME):
    # Outputting graphs for Accuracy
    plt.figure()
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model train_accuracy vs val_accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(os.path.join(newdir_path, a_plot_name))
    plt.close()

    # Outputting graphs for Loss
    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model train_loss vs val_loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(os.path.join(newdir_path, l_plot_name))
    plt.close()

def get_heatmap(model, X_test, y_test, newdir_path, hm_name, label_list):
    prediction = model.predict(X_test)
    y_pred = np.argmax(prediction, axis=1)

    labels = sorted(label_list)  # Sort the labels
    column = [f'Predicted {label}' for label in labels]
    indices = [f'Actual {label}' for label in labels]
    table = pd.DataFrame(confusion_matrix(y_test, y_pred), columns=column, index=indices)

    plt.figure()
    hm = sns.heatmap(table, annot=True, fmt='d', cmap='viridis')
    plt.savefig(os.path.join(newdir_path, hm_name))
    plt.close()
    print("Heatmap generated and saved in {path}".format(path=NEWDIR_PATH))

def prepare_datasets(test_size, validation_size):

    # load data
    X, y, label_list = load_data(DATA_PATH)

    # create train, validation, and test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=validation_size)

    return X_train, X_validation, X_test, y_train, y_validation, y_test, label_list

def Parallel_CNN_RNN(input_shape, num_classes):
    input = keras.layers.Input(shape=input_shape)
    
    C_layer = keras.layers.Conv2D(filters=16, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same')(input)
    C_layer = keras.layers.BatchNormalization()(C_layer)
    C_layer = keras.layers.Dropout(0.2)(C_layer)
    C_layer = keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2))(C_layer)
    
    C_layer = keras.layers.Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same')(C_layer)
    C_layer = keras.layers.BatchNormalization()(C_layer)
    C_layer = keras.layers.Dropout(0.2)(C_layer)
    C_layer = keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2))(C_layer)
    
    C_layer = keras.layers.Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same')(C_layer)
    C_layer = keras.layers.BatchNormalization()(C_layer)
    C_layer = keras.layers.Dropout(0.2)(C_layer)
    C_layer = keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2))(C_layer)
    
    # C_layer = keras.layers.Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same')(C_layer)
    # C_layer = keras.layers.BatchNormalization()(C_layer)
    # C_layer = keras.layers.Dropout(0.2)(C_layer)
    # C_layer = keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2))(C_layer)
    
    # C_layer = keras.layers.Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same')(C_layer)
    # C_layer = keras.layers.BatchNormalization()(C_layer)
    # C_layer = keras.layers.Dropout(0.2)(C_layer)
    # C_layer = keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2))(C_layer)
    C_layer = keras.layers.Flatten()(C_layer)
    
    # R_layer = keras.layers.MaxPooling2D(pool_size=(1,1), strides=(1,2))(input)
    # R_layer = keras.layers.MaxPooling2D(pool_size=(1,2), strides=(1,2))(R_layer)
    R_layer = keras.layers.Reshape(target_shape=(130,13))(input)
    # R_layer = keras.layers.Embedding(input_dim=256, output_dim=128, input_length=128)(R_layer)
    R_layer = keras.layers.Dense(128, activation='relu')(R_layer)
    # R_layer = keras.layers.Reshape(target_shape=(128, 128))(R_layer)
    # R_layer = keras.layers.Embedding(input_dim=256, output_dim=128)(R_layer)
    R_layer = keras.layers.Bidirectional(keras.layers.GRU(512, return_sequences=True))(R_layer)
    R_layer = keras.layers.BatchNormalization()(R_layer)
    R_layer = keras.layers.Dropout(0.5)(R_layer)
    R_layer = keras.layers.Bidirectional(keras.layers.GRU(512, return_sequences=False))(R_layer)
    R_layer = keras.layers.BatchNormalization()(R_layer)
    R_layer = keras.layers.Dropout(0.5)(R_layer)
    # R_layer5 = keras.layers.Bidirectional(keras.layers.GRU(256))(R_layer4)
    
    concat = keras.layers.concatenate([C_layer, R_layer], axis=1)
    output = keras.layers.Dense(10, activation='softmax')(concat)
    
    model = keras.Model(inputs=[input], outputs=[output])
    
    
    return model

if __name__ == "__main__":
    # create train, val, test sets
    X_train, X_validation, X_test, y_train, y_validation, y_test, label_list = prepare_datasets(0.25, 0.2)
    X_train = X_train[..., np.newaxis]    #4d array -> (num_samples, 130, 13, 1)
    X_validation = X_validation[..., np.newaxis]
    X_test = X_test[..., np.newaxis]
    
    
    # Define the input shapes and number of classes
    input_shape = (X_train.shape[1], X_train.shape[2], 1)  # Assumes input audio features of shape (num_timesteps, num_features)
    num_classes = NUM_CLASSES  # Number of music genres

    # Create the combined model
    model = Parallel_CNN_RNN(input_shape, num_classes)
    
    optimiser = keras.optimizers.Adam(learning_rate=LEARNING_RATE)

    # Compile the model
    model.compile(optimizer=optimiser,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Print the model summary
    model.summary()
    
    """
    # Drop-Based Learning Rate Schedule
    lr_scheduler = keras.callbacks.LearningRateScheduler(step_decay)
    # Train the model
    history = model.fit(X_train, y_train, validation_data=(X_validation, y_validation),
                        batch_size=32, epochs=EPOCHS, verbose=1, callbacks=[lr_scheduler])
    """
    
    # Train the model
    history = model.fit(X_train, y_train, validation_data=(X_validation, y_validation),
                        batch_size=32, epochs=EPOCHS, verbose=1)
    
    print("Finished Training Model!")
    
    # Print validation loss and accuracy
    val_loss, val_acc = model.evaluate(X_test, y_test)
    print("Valdiation Loss: ", val_loss)
    print("Valdiation Accuracy: ", val_acc)

    # Plot history
    save_plot(history)

    # Evaluate model on test set
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
    print('\nTest accuracy:', test_acc)

    # Pick a sample to predict from the test set
    X_to_predict = X_test[10]
    y_to_predict = y_test[10]

    # Predict sample
    #predict(model, X_to_predict, y_to_predict)

    # Save model
    if SAVE_MODEL:
        model.save(os.path.join(NEWDIR_PATH, MODEL_NAME))
        print("Model saved to disk at:", os.path.join(NEWDIR_PATH, MODEL_NAME))

    # Output heatmap
    if SAVE_HM:
        get_heatmap(model, X_test, y_test, NEWDIR_PATH, HM_NAME, label_list)
