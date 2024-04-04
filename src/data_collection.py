import pickle
import numpy as np
import pandas as pd

def create_train_test_datasets(train_dataset_path, test_dataset_path):
    '''
        Create train and test dataframes.

        Args:
            train_dataset_path: path of the train dataset
            test_dataset_path: path of the test dataset

        Return:
            train_df: Final train dataframe
            test_df: Final test dataframe
    '''

    # ================== Train Dataset ============================

    # Read train dataframe
    with open(train_dataset_path, 'rb') as f:
        train_df = pickle.load(f)

    # Rename columns
    train_df.rename(columns={"label": "Label", "text": "Domain Name"}, inplace=True)

    # Create a new column with the family of the data sample.
    train_df["Family"] = train_df["Label"]

    # Re-order columns
    train_df = train_df.reindex(columns=["Domain Name", "Family", "Label"])

    # Convert Label column with values 0(legit) and 1(dga).
    train_df['Label'] = train_df['Label'].apply(lambda x: 0 if x == 'benign' else 1)

    # Shuffle the rows
    train_df = train_df.sample(frac=1, random_state=42)

    # Reset index
    train_df.reset_index(drop=True, inplace=True)

    # ================== Test Dataset ============================

    # Read test dataframe
    test_df = pd.read_csv(test_dataset_path, header=None)

    # Rename columns.
    test_df.rename(columns={0: "Label", 1: "Family", 2: "Domain Name"}, inplace=True)

    # Re-order columns
    test_df = test_df.reindex(columns=["Domain Name", "Family", "Label"])

    # Convert Label column with values 0(legit) and 1(dga).
    test_df['Label'] = test_df['Label'].apply(lambda x: 0 if x == 'legit' else 1)

    # Convert in Family column every alexa value to benign
    test_df['Family'] = test_df['Family'].apply(lambda x: 'benign' if x == 'alexa' else x)

    # Keep only the families that belong to the train dataset.
    test_df = test_df[test_df['Family'].isin(np.unique(train_df['Family']))]

    # Shuffle the rows
    test_df = test_df.sample(frac=1, random_state=42)

    # Reset index
    test_df.reset_index(drop=True, inplace=True)

    # ===============================================================

    print("Train dataset:\n")
    print_df_stats(train_df)

    print("Test dataset:\n")
    print_df_stats(test_df)

    return train_df, test_df

def print_df_stats(df):
    '''
        Print number of data samples in each category and in each family.
    '''
    print(df)
    print(df['Label'].value_counts())
    print(df['Family'].value_counts())

if __name__ == "__main__":
    # Fix path
    train_dataset_path = r"C:\Users\plust\OneDrive\Desktop\Jupyter Projects\MSc - DSML\Thesis\data\original_train_data.pkl"
    test_dataset_path = r"C:\Users\plust\OneDrive\Desktop\Jupyter Projects\MSc - DSML\Thesis\data\original_test_data.csv"
    total_data_folder_path = r"C:\Users\plust\OneDrive\Desktop\Jupyter Projects\MSc - DSML\Thesis\data"

    # Create train and test dataset.
    train_df, test_df = create_train_test_datasets(train_dataset_path, test_dataset_path)

    # Save final train and test datasets to a csv file.
    train_df.to_csv(path_or_buf=total_data_folder_path + "/train_df.csv", sep=",", index=False, header=True, mode="w")
    test_df.to_csv(path_or_buf=total_data_folder_path + "/test_df.csv", sep=",", index=False, header=True, mode="w")