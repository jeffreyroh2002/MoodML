import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from sklearn.model_selection import KFold
import tensorflow as tf
import tensorflow.keras as keras
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import pickle
from keras.callbacks import LearningRateScheduler

####EDIT BEFORE RUNNING ###########
# path to json file that stores MFCCs and genre labels for each processed segment
DATA_PATH = "../../audio_file/preprocessed/STFT_GTZAN_dataset.txt"
SAVE_MODEL = True
SAVE_HM = True

#OUTPUT DIR/FILE NAMES
NEWDIR_NAME = "genre_PRCNN-0804"

MODEL_NAME = "saved_model"
HM_NAME = "heatmap3.png"
A_PLOT_NAME = 'accuracy3.png'
L_PLOT_NAME = 'loss3.png'

# Hyperparameters
LEARNING_RATE = 0.002
EPOCHS = 100

####################################

#create new dir in results dir for results
NEWDIR_PATH = os.path.join("../../results", NEWDIR_NAME)
if not os.path.exists(NEWDIR_PATH):
    os.makedirs(NEWDIR_PATH)

def linearly_decreasing_lr(epoch, current_lr):
    max_epochs = 100  # Total number of epochs
    initial_lr = 0.002  # Starting learning rate
    final_lr = 0.0  # Final learning rate
    
    if epoch >= max_epochs:
        return final_lr
    new_lr = initial_lr - (epoch / max_epochs) * (initial_lr - final_lr)
    return new_lr    
    
    
def load_data(data_path):
    with open(data_path, "rb") as fp:
        data = pickle.load(fp)

    # convert lists to numpy arrays
    X = np.array(data["stft"])
    y = np.array(data["labels"])
    label_list = data.get("mapping", {})   #Jazz, Classical, etc

    print(label_list)

    print("Data successfully loaded!")

    return X, y, label_list

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

# def prepare_cnn_datasets(test_size, validation_size):
#     # load data
#     X, y, label_list = load_data(DATA_PATH)

#     # create train, validation, and test split
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
#     X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=validation_size)

    
#     # add an axis to input sets (CNN requires 3D array)
#     X_train = X_train[..., np.newaxis]    #4d array -> (num_samples, 130, 13, 1)
    # X_validation = X_validation[..., np.newaxis]
#     X_test = X_test[..., np.newaxis]

#     return X_train, X_validation, X_test, y_train, y_validation, y_test, label_list
    
def prepare_datasets(test_size, validation_size):

    # load data
    X, y, label_list = load_data(DATA_PATH)

    # create train, validation, and test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=validation_size)

    return X_train, X_validation, X_test, y_train, y_validation, y_test, label_list

# def create_combined_model(input_shape, num_classes):
#     input = keras.layers.Input(shape=input_shape)
#     C_layer1 = keras.layers.Conv2D(filters=16, kernel_size=(3,1), strides=(1,1), activation='relu', padding='same')(input)
#     C_layer2 = keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2))(C_layer1)
#     C_layer3 = keras.layers.Conv2D(filters=32, kernel_size=(3,1), strides=(1,1), activation='relu', padding='same')(C_layer2)
#     C_layer4 = keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2))(C_layer3)
#     C_layer5 = keras.layers.Conv2D(filters=64, kernel_size=(3,1), strides=(1,1), activation='relu', padding='same')(C_layer4)
#     C_layer6 = keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2))(C_layer5)
#     C_layer7 = keras.layers.Conv2D(filters=128, kernel_size=(3,1), strides=(1,1), activation='relu', padding='same')(C_layer6)
#     C_layer8 = keras.layers.MaxPooling2D(pool_size=(4,4), strides=(4,4))(C_layer7)
#     C_layer9 = keras.layers.Conv2D(filters=64, kernel_size=(3,1), strides=(1,1), activation='relu', padding='same')(C_layer8)
#     C_layer10 = keras.layers.MaxPooling2D(pool_size=(4,4), strides=(4,4))(C_layer9)
    
#     R_layer1 = keras.layers.MaxPooling2D(pool_size=(1,2), strides=(1,2))(input)
#     R_layer2 = keras.layers.Embedding(input_dim=256, output_dim=128)(R_layer1)
#     R_layer3 = keras.layers.Bidirectional(keras.layers.GRU(256))(R_layer2)
#     R_layer4 = keras.layers.Bidirectional(keras.layers.GRU(256))(R_layer3)
    
#     concat = tf.layers.concatenate([C_layer10, R_layer4], axis=1)
#     output = keras.layers.Dense(10, activation='softmax')(concat)
    
#     model = keras.Model(inputs=[input], outputs=[output])
    
def Parallel_CNN_RNN(input_shape):
    input = keras.layers.Input(shape=input_shape)
    
    C_layer1 = keras.layers.Conv2D(filters=16, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same')(input)
    C_layer1 = keras.layers.BatchNormalization()(C_layer1)
    C_layer1 = keras.layers.Dropout(0.2)(C_layer1)
    C_layer2 = keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2))(C_layer1)
    
    C_layer3 = keras.layers.Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same')(C_layer2)
    C_layer3 = keras.layers.BatchNormalization()(C_layer3)
    C_layer3 = keras.layers.Dropout(0.2)(C_layer3)
    C_layer4 = keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2))(C_layer3)
    
    C_layer5 = keras.layers.Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same')(C_layer4)
    C_layer5 = keras.layers.BatchNormalization()(C_layer5)
    C_layer5 = keras.layers.Dropout(0.2)(C_layer5)
    C_layer6 = keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2))(C_layer5)
    
    C_layer7 = keras.layers.Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same')(C_layer6)
    C_layer7 = keras.layers.BatchNormalization()(C_layer7)
    C_layer7 = keras.layers.Dropout(0.2)(C_layer7)
    C_layer8 = keras.layers.MaxPooling2D(pool_size=(4,4), strides=(4,4))(C_layer7)
    
    C_layer9 = keras.layers.Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same')(C_layer8)
    C_layer9 = keras.layers.BatchNormalization()(C_layer9)
    C_layer9 = keras.layers.Dropout(0.2)(C_layer9)
    C_layer10 = keras.layers.MaxPooling2D(pool_size=(4,4), strides=(4,4))(C_layer9)
    C_layer11 = keras.layers.Flatten()(C_layer10)
    
    R_layer1 = keras.layers.MaxPooling2D(pool_size=(1,2), strides=(1,2))(input)
    # R_layer2 = keras.layers.MaxPooling2D(pool_size=(1,2), strides=(1,2))(R_layer1)
    R_layer2 = keras.layers.Reshape(target_shape=(128,256))(R_layer1)
    # R_layer3 = keras.layers.Embedding(input_dim=256, output_dim=128, input_length=128)(R_layer2)
    R_layer3 = keras.layers.Dense(128, activation='relu')(R_layer2)
    # R_layer4 = keras.layers.Reshape(target_shape=(128, 128))(R_layer3)
    # R_layer2 = keras.layers.Embedding(input_dim=256, output_dim=128)(R_layer1)
    R_layer4 = keras.layers.Bidirectional(keras.layers.GRU(128, return_sequences=True))(R_layer3)
    R_layer4 = keras.layers.BatchNormalization()(R_layer4)
    R_layer4 = keras.layers.Dropout(0.5)(R_layer4)
    R_layer4 = keras.layers.Bidirectional(keras.layers.GRU(128, return_sequences=False))(R_layer4)
    R_layer4 = keras.layers.BatchNormalization()(R_layer4)
    R_layer4 = keras.layers.Dropout(0.5)(R_layer4)
    # R_layer5 = keras.layers.Bidirectional(keras.layers.GRU(256))(R_layer4)
    
    concat = keras.layers.concatenate([C_layer11, R_layer4], axis=1)
    output = keras.layers.Dense(10, activation='softmax')(concat)
    
    model = keras.Model(inputs=[input], outputs=[output])
    
    
    return model
    
"""
def create_cnn(input_shape):
    cnn_model = keras.Sequential()

    cnn_model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    cnn_model.add(keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'))
    cnn_model.add(keras.layers.BatchNormalization())

    cnn_model.add(keras.layers.Conv2D(32, (3, 3), activation='relu'))
    cnn_model.add(keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'))
    cnn_model.add(keras.layers.BatchNormalization())

    cnn_model.add(keras.layers.Conv2D(32, (3, 3), activation='relu'))
    cnn_model.add(keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'))
    cnn_model.add(keras.layers.BatchNormalization())

    cnn_model.add(keras.layers.Flatten())
    return cnn_model

# Define the bi-directional RNN model
def create_rnn(input_shape):
    rnn_model = keras.Sequential()
    rnn_model.add(keras.layers.Bidirectional(keras.layers.GRU(128, return_sequences=True), input_shape=input_shape))
    rnn_model.add(keras.layers.Bidirectional(keras.layers.GRU(128)))
    return rnn_model



# Define the combined model
def create_combined_model(cnn_input_shape, rnn_input_shape, num_classes):
    cnn_model = create_cnn(cnn_input_shape)
    rnn_model = create_rnn(rnn_input_shape)

    combined_model = keras.Sequential()
    combined_model.add(keras.layers.concatenate([cnn_model.output, rnn_model.output]))
    combined_model.add(keras.layers.Dense(128, activation='relu'))
    combined_model.add(keras.layers.Dense(num_classes, activation='softmax'))

    return combined_model
"""

def predict(model, X, y):
    """Predict a single sample using the trained model

    :param model: Trained classifier
    :param X: Input data
    :param y (int): Target
    """
    # add a dimension to input data for sample - model.predict() expects a 4d array in this case

    X = X[np.newaxis, ...] # array shape (1, 130, 13, 1)

    # perform prediction
    prediction = model.predict(X)

    # get index with max value
    predicted_index = np.argmax(prediction, axis=1)

    print("Target: {}, Predicted label: {}".format(y, predicted_index))

    
if __name__ == "__main__":
    # create train, val, test sets
    X_train, X_validation, X_test, y_train, y_validation, y_test, label_list = prepare_datasets(0.2, 0.2)
    # X_train,  X_test, y_train,  y_test, label_list = prepare_datasets(0.2, 0)
    # add an axis to input sets (CNN requires 3D array)
    X_train = X_train[..., np.newaxis]    #4d array -> (num_samples, 130, 13, 1)
    X_validation = X_validation[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

    # Set random seed for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)

    # Define the input shapes and number of classes
    input_shape = (X_train.shape[1], X_train.shape[2], 1) # Assumes input audiofeatures of shape (num_timesteps, num_features)
    # # Define the hidden size for LSTM layers
    # hidden_size = 64
    
    num_classes = 10  # Number of music genres
    
    # Create the combined model
    model = Parallel_CNN_RNN(input_shape)

    optimiser = keras.optimizers.Adam(0.001)

    # Compile the model
    model.compile(optimizer=optimiser,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Print the model summary
    model.summary()
    
    lr_scheduler = LearningRateScheduler(linearly_decreasing_lr)

    # model.fit(x_train, y_train, epochs=max_epochs, callbacks=[lr_scheduler])
    
    # kf = KFold(n_splits=5)
    # for train_index, val_index in kf.split(X_train):

    #     kf_X_train = X_train[train_index]
    #     kf_X_val = X_train[val_index]
    #     kf_y_train = y_train[train_index]
    #     kf_y_val = y_train[val_index]

    #     history = model.fit(kf_X_train, kf_y_train, validation_data=(kf_X_val, kf_y_val), epochs=100, batch_size=32, callbacks=[lr_scheduler], verbose=1)


    # Train the model
    history = model.fit(X_train, y_train, validation_data=(X_validation, y_validation),
                        batch_size=32, epochs=EPOCHS, verbose=1)
    
    print("Finished Training Model!")

    
    # # Print validation loss and accuracy
    # val_loss, val_acc = model.evaluate(X_validation, y_validation)
    # print("Validation Loss:", val_loss)
    # print("Validation Accuracy:", val_acc)

    # Plot history
    save_plot(history)

    # Evaluate model on test set
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
    print('\nTest accuracy:', test_acc)

    # # Pick a sample to predict from the test set
    # X_to_predict = cnn_X_test[10]
    # y_to_predict = cnn_y_test[10]

    # Predict sample
    #predict(model, X_to_predict, y_to_predict)

    # Save model
    if SAVE_MODEL:
        model.save(os.path.join(NEWDIR_PATH, MODEL_NAME))
        print("Model saved to disk at:", os.path.join(NEWDIR_PATH, MODEL_NAME))

    # Output heatmap
    if SAVE_HM:
        get_heatmap(model, X_test, y_test, NEWDIR_PATH, HM_NAME, label_list)