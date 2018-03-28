import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os


model_name = input('model name: ')
all_data = pd.read_csv('train_merged.csv', sep=',', header='infer',
                    dtype={'Store':int, 'Dept':str, 'IsHoliday':str})

del all_data['MarkDown1']
del all_data['MarkDown2']
del all_data['MarkDown3']
del all_data['MarkDown4']
del all_data['MarkDown5']
del all_data['Date']
del all_data['Type']
del all_data['Size']
all_data = pd.get_dummies(all_data)

def get_data_store(num):

    data = all_data[all_data['Store']==int(num)]

    Y = data['Weekly_Sales']
    del data['Weekly_Sales']
    del data['Store']

    X = data.values
    Y = Y.values
    X = X.astype('float32')
    Y = Y.astype('float32')

    return train_test_split(X, Y, test_size=0.1, random_state=13)


from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import h5py


EPOCHS = int(input('Epochs :'))


def build_model(input_shape):
    model = Sequential()
    model.add(Dense(64, input_shape=input_shape, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer=Adam(lr=0.01), loss='mean_absolute_error')
    return model

def evaluate(model, x, y):
    predicted = model.predict(x)[:, 0]
    error = np.absolute(predicted - y)
    mae = sum(error) / len(error)
    return {'predicted':predicted, 'mae':mae}

models = []
for i in range(45):
    X_train, X_test, Y_train, Y_test = get_data_store(i+1)
    models.append(build_model(input_shape=(X_train.shape[1],)))
    history = models[i].fit(X_train, Y_train,
            epochs=EPOCHS,
            batch_size=X_train.shape[0],
            verbose=2)

    print(i+1, ' MAE ', evaluate(models[i], X_test, Y_test)['mae'])
    print()

    if not os.path.exists('models/'+model_name):
        os.makedirs('models/'+model_name)
    models[i].save('models/'+model_name+'/s'+str(i+1)+'.h5py')
