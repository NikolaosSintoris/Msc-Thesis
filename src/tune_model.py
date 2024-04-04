import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Flatten,Dropout
from tensorflow import keras
from keras_tuner.tuners import RandomSearch, BayesianOptimization

def create_model(n_features):
    '''
        Create an MLP model.
    '''
    model = Sequential([
        Dense(units=128, input_shape=(n_features,), activation='relu'),
        Dense(units=32, activation='relu'),
        Dense(units=16, activation='relu'),
        Dense(units=1, activation='sigmoid')
    ])

    # Compile model.
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

def create_supervised_train_test_sets(train_df, test_df):
    '''
        Create X train, y train and X test, y test.
    '''

    # Drop domain name, family and Label column to create X_train data.
    X_train = np.array(train_df.drop(["Domain Name", "Family", "Label"], axis=1))

    # Create a y np array with the labels.
    y_train = np.array(train_df["Label"])

    # Drop domain name, family and Label column to create X_test data.
    X_test = np.array(test_df.drop(["Domain Name", "Family", "Label"], axis=1))

    # Create a y np array with the labels.
    y_test = np.array(test_df["Label"])

    # Normalize datasets.
    X_train, X_test = MinMax_normalization(X_train, X_test)

    return X_train, y_train, X_test, y_test

def MinMax_normalization(X_train, X_test):
    '''
        Normalize data using MinMax Normalization

            Input:
                Train, validation and test set

            Return:
                Scaled train, validation and test set
    '''

    # Create a scaler based on train dataset.
    scaler_obj = MinMaxScaler()
    X_train_scaled = scaler_obj.fit_transform(X_train)

    # Transform validation and test set based on the training scaler.
    X_test_scaled = scaler_obj.transform(X_test)

    return X_train_scaled, X_test_scaled

def build_model(hp, X_train):
    model = Sequential()
    model.add(Flatten(input_shape=(X_train.shape[1],)))

    # Tune the number of Dense layers
    for i in range(hp.Int('num_layers', min_value=1, max_value=2)):  # Adjust max_value as needed
        model.add(Dense(units=hp.Int('units_' + str(i), min_value=32, max_value=128, step=32), activation='relu'))

        # Add a dropout layer if selected
        if hp.Boolean('use_dropout'):
            model.add(Dropout(rate=hp.Float('dropout_rate', min_value=0.2, max_value=0.5)))

    model.add(Dense(units=1, activation='sigmoid'))  # Output layer for classification (adjust as needed)

    # Compile the model
    model.compile(optimizer=keras.optimizers.Adam(hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])), \
                  loss='binary_crossentropy', \
                  metrics=['accuracy'])

    return model

if __name__ == '__main__':
    # Fix path
    final_train_path = "/kaggle/input/non-federated-train-test-df/final_train_df.csv"
    final_test_path = "/kaggle/input/non-federated-train-test-df/final_test_df.csv"

    # Load final train and test dataframes.
    train_df = pd.read_csv(final_train_path, header=0)
    test_df = pd.read_csv(final_test_path, header=0)

    # Create X and y arrays, for train and test sets.
    X_train, y_train, X_test, y_test = create_supervised_train_test_sets(train_df, test_df)


    tuner = BayesianOptimization(build_model, \
                                 objective='val_accuracy', \
                                 max_trials=10, \
                                 directory='my_tuner_dir', \
                                 project_name='my_mlp_tuning_1')

    tuner.search(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

    best_model = tuner.get_best_models(num_models=1)[0]
    print(best_model.summary())

    best_hparams = tuner.get_best_hyperparameters(num_trials=1)[0]
    print("Number of layers: {}".format(best_hparams.get('num_layers')))
    print("Number of units_0: {}".format(best_hparams.get('units_0')))
    # print("Number of units_1: {}".format(best_hparams.get('units_1')))
    # print("Number of units_2: {}".format(best_hparams.get('units_2')))
    print("Use dropout: {}".format(best_hparams.get('use_dropout')))
    # print("Dropout Rate: {}".format(best_hparams.get('dropout_rate')))
    print("Learning Rate: {}".format(best_hparams.get('learning_rate')))
