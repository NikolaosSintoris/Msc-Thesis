import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt

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


def k_fold_training_results_line_plot(train_results_dict):
    # Create a figure with a 3x2 grid of subplots
    fig, axs = plt.subplots(3, 2, figsize=(25, 10))  # You can adjust the figsize as needed

    # Flatten the axs array for easier iteration
    axs = axs.flatten()

    # Create subplots with two lines in each subplot
    for i, ax in enumerate(axs):
        ax.plot(train_results_dict[str(i)][0], label='training loss')
        ax.plot(train_results_dict[str(i)][1], label='Validation loss')
        ax.set_title(f'Fold {i + 1}')
        ax.legend()

    # Adjust the layout and spacing
    plt.tight_layout()

    plt.savefig('non_federated_mlp_loss_plot_kfold.png')

    # Show the plots
    plt.show()

if __name__ == '__main__':
    # Fix path
    final_train_path = "/kaggle/input/non-federated-train-test-df/final_train_df.csv"
    final_test_path = "/kaggle/input/non-federated-train-test-df/final_test_df.csv"

    # Load final train and test dataframes.
    train_df = pd.read_csv(final_train_path, header=0)
    test_df = pd.read_csv(final_test_path, header=0)

    # Create X and y arrays, for train and test sets.
    X_train, y_train, X_test, y_test = create_supervised_train_test_sets(train_df, test_df)

    no_folds = 6
    skf_obj = StratifiedKFold(n_splits=no_folds, shuffle=True, random_state=42)

    train_results_dict = {}
    evaluation_results_dict = {}
    for i, (train_index, test_index) in enumerate(skf_obj.split(X_train, y_train)):
        print('Fold: {}'.format(i))

        # Create current train and validation sets.
        current_X_train = X_train[train_index]
        current_y_train = y_train[train_index]

        X_validation = X_train[test_index]
        y_validation = y_train[test_index]

        # Create MLP model.
        n_features = current_X_train.shape[1]
        mlp = create_model(n_features)

        # Train MLP on the dataset
        history = mlp.fit(current_X_train, current_y_train, epochs=100, batch_size=32,
                          validation_data=(X_validation, y_validation), verbose=1)

        # Save training results: training and validation loss.
        train_results_dict[str(i)] = [history.history['loss'], history.history["val_loss"]]

        # Get the predictions of the model.
        y_pred = mlp.predict(X_test)
        y_pred = y_pred.flatten()
        y_pred = np.round(y_pred)
        y_pred = y_pred.astype(int)

        # Save evaluation results: Accuracy, Precision, Recall, F1-score.
        evaluation_results_dict[str(i)] = [accuracy_score(y_test, y_pred), \
                                           precision_score(y_test, y_pred), \
                                           recall_score(y_test, y_pred), \
                                           f1_score(y_test, y_pred) \

                                           ]
    # Plot train and validation loss plot.
    k_fold_training_results_line_plot(train_results_dict)

    # Print a dataframe with the results on test set for every folder.
    evaluation_results_df = pd.DataFrame(evaluation_results_dict)
    evaluation_results_df = evaluation_results_df.transpose()
    evaluation_results_df.rename(columns={0: 'Accuracy', 1: 'Precision', 2: 'Recall', 3: 'F1-score'}, inplace=True)
    print(evaluation_results_df)