import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

import os
import collections

def construct_model (input_shape):
    model = Sequential()
    model.add(Dense(64, input_shape=input_shape, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer=Adam(lr=0.01), loss='mean_absolute_error')

    return model

def predict (model, valid_X, valid_Y):
    predicted = model.predict(valid_X)

    error = np.zeros(valid_Y.size)
    for i in range(valid_Y.size):
        error[i] = abs(predicted[i] - valid_Y[i])
    mae = sum(error) / len(error)
    return {'predicted':predicted, 'mae':mae}

def get_data (test_size = 0.2):
    features = ['Store', 'Dept', 'Fuel_Price', 'CPI', 'Unemployment', 'Type', 'Size']

    file_data = pd.read_csv('../train_merged.csv')

    X = file_data[features]
    X = pd.get_dummies(X)
    X = X.values

    Y = file_data['Weekly_Sales']
    Y = Y.values

    for x, y in zip(X, Y):
        x = x.astype('float32')
        y = y.astype('float32')

    train_X, valid_X, train_Y, valid_Y = train_test_split(X, Y, test_size=test_size)
    return X, Y, train_X, valid_X, train_Y, valid_Y

def get_model (model_name):
    return load_model('../models/'+model_name+'.mod')

#--------------------------------------------------------------------------------------------------#

def main():
    X, Y, train_X, valid_X, train_Y, valid_Y = get_data()

    if str(input('Load model? [y/n]: ')) == 'y':
        model_name = str(input('Model name: '))
        model = get_model(model_name)

        print(' MAE ', predict(model, valid_X, valid_Y)['mae'])

    else:
        batch_size = int(input('Batch size [len(Y)]: ') or len(Y))
        epochs = int(input('Epochs [20]: ') or 20)
        input_shape = X[0].shape

        model = construct_model(input_shape)

        model.fit(
                train_X,
                train_Y,
                epochs=epochs,
                batch_size=batch_size,
                verbose=2)

        print(' MAE ', predict(model, valid_X, valid_Y)['mae'])

        if not os.path.exists("../models"):
            os.makedirs("../models")

        if str(input('Save model? [y/n]: ')) == 'y':
            model_name = str(input('Model name: '))
            model.save('../models/'+model_name+'.mod')

#--------------------------------------------------------------------------------------------------#

if __name__ == "__main__":
    main()
