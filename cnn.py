import keras
from keras import backend as K
from keras.models import Sequential,Input,Model
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'



def simple_model(input_size, output_size ):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu',input_shape=(input_size, input_size, 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Flatten())
    model.add(Dense(96, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(output_size, activation='sigmoid'))
    return model


def fit_model( train_X, train_Y,test_X, test_Y, input_size = 96, output_size = 39, **kwargs ):
    lr = kwargs.get('lr', None)
    epochs = kwargs.get('epochs', None)
    batch_size = kwargs.get('batch_size', None)
    
    opt =  keras.optimizers.Adam(learning_rate=lr, decay= lr / epochs)
    model = simple_model(input_size, output_size )
    model.compile(loss= "binary_crossentropy", optimizer=opt, metrics=[f1_keras])
    
    history = model.fit(train_X, train_Y, epochs=epochs, batch_size = batch_size, validation_data=(test_X, test_Y))
    return history

def recall_keras(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_keras(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_keras(y_true, y_pred):
    precision = precision_keras(y_true, y_pred)
    recall = recall_keras(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))



def visualize_history(history, metric):
    try:
        plt.plot(history.history[metric])
        plt.plot(history.history[f'val_{metric}'])
        plt.title(f'Model {metric}')
        plt.ylabel(metric)
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
    except KeyError:
        raise KeyError ('As a metric choose among:', list(history.history.keys()))