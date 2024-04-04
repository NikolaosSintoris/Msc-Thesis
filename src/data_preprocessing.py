import numpy as np
import pandas as pd
import re

def keep_domain_name(full_domain_name):
    domain_parts_lst = full_domain_name.split(".")

    return domain_parts_lst[0]

def domain_name_length(my_string):
    '''
        Find the length of a string
    '''
    return len(my_string)

def consonants_frequency(my_string):
    '''
        Find number of consonants in a string
    '''
    consonant_pattern = re.compile(r'[BCDFGHJKLMNPQRSTVWXZ]', re.IGNORECASE)
    return len(re.findall(consonant_pattern, my_string))

def max_consonants_sequence_length(my_string):
    '''
        Find the length of the maximum consonants sequence in a string
    '''
    consonant_sequences_lst = re.findall(r'[BCDFGHJKLMNPQRSTVWXZ]+', my_string.upper())

    if (len(consonant_sequences_lst) == 0):
        return 0

    return len(max(consonant_sequences_lst, key=len))

def min_consonants_sequence_length(my_string):
    '''
        Find the length of the minimum consonants sequence in a string
    '''
    consonant_sequences_lst = re.findall(r'[BCDFGHJKLMNPQRSTVWXZ]+', my_string.upper())

    if (len(consonant_sequences_lst) == 0):
        return 0

    return len(min(consonant_sequences_lst, key=len))

def vowels_frequency(my_string):
    '''
        Find number of vowels in a string
    '''
    vowels = r'[aeiouyAEIOUY]'
    return len(re.findall(vowels, my_string))

def max_vowels_sequence_length(my_string):
    '''
        Find the length of the maximum vowels sequence in a string
    '''
    vowels_sequences_lst = re.findall(r'[aeiouyAEIOUY]+', my_string.upper())

    if (len(vowels_sequences_lst) == 0):
        return 0

    return len(max(vowels_sequences_lst, key=len))

def min_vowels_sequence_length(my_string):
    '''
        Find the length of the minimum vowels sequence in a string
    '''
    vowels_sequences_lst = re.findall(r'[aeiouyAEIOUY]+', my_string.upper())

    if (len(vowels_sequences_lst) == 0):
        return 0

    return len(min(vowels_sequences_lst, key=len))

def digits_frequency(my_string):
    '''
        Find number of digits in a string
    '''
    return len(re.findall('[0-9]', my_string))

def max_digits_sequence_length(my_string):
    '''
        Find the length of the maximum digits sequence in a string
    '''
    digits_sequences_lst = re.findall('[0-9]+', my_string)

    if (len(digits_sequences_lst) == 0):
        return 0

    return len(max(digits_sequences_lst, key=len))

def min_digits_sequence_length(my_string):
    '''
        Find the length of the minimum digits sequence in a string
    '''
    digits_sequences_lst = re.findall(r'[0-9]+', my_string.upper())

    if (len(digits_sequences_lst) == 0):
        return 0
    return len(min(digits_sequences_lst, key=len))

def hyphen_frequency(my_string):
    '''
        Find number of hyphens in a string
    '''
    hyphen_pattern = re.compile(r'-')
    return len(re.findall(hyphen_pattern, my_string))

def max_hyphen_sequence_length(my_string):
    '''
        Find the length of the maximum hyphens sequence in a string
    '''
    hyphen_sequences_lst = re.findall(r'[-]+', my_string)

    if (len(hyphen_sequences_lst) == 0):
        return 0

    return len(max(hyphen_sequences_lst, key=len))

def min_hyphen_sequence_length(my_string):
    '''
        Find the length of the minimum hyphens sequence in a string
    '''
    hyphen_sequences_lst = re.findall(r'[-]+', my_string)

    if (len(hyphen_sequences_lst) == 0):
        return 0

    return len(min(hyphen_sequences_lst, key=len))

def max_letter_sequence_length(my_string):
    '''
        Find the length of the maximum letters sequence in a string
    '''
    letter_sequences_lst = re.findall(r'[a-zA-Z]+', my_string)

    if (len(letter_sequences_lst) == 0):
        return 0

    return len(max(letter_sequences_lst, key=len))

def min_letter_sequence_length(my_string):
    '''
        Find the length of the minimum letters sequence in a string
    '''
    letter_sequences_lst = re.findall(r'[a-zA-Z]+', my_string)

    if (len(letter_sequences_lst) == 0):
        return 0

    return len(min(letter_sequences_lst, key=len))

def character_frequency(my_string, character):
    '''
        Find the number of a character appears in a string
    '''
    character = character.lower()
    return my_string.lower().count(character)

def feature_extractor(df):
    '''
        Create final dataset with the new statistical features.

        Args:
            df: Init dataframe

        Return:
            df: Dataframe after feature extractor with new features
    '''

    df['Domain Name Length'] = df.drop(['Family', 'Label'], axis=1).apply(
        lambda row: domain_name_length(str(row['Domain Name'])), axis=1)

    df['Consonants Freq'] = df.drop(['Family', 'Label'], axis=1).apply(
        lambda row: consonants_frequency(str(row['Domain Name'])), axis=1)

    df['Max Consonants Seq Length'] = df.drop(['Family', 'Label'], axis=1).apply(
        lambda row: max_consonants_sequence_length(str(row['Domain Name'])), axis=1)

    df['Min Consonants Seq Length'] = df.drop(['Family', 'Label'], axis=1).apply(
        lambda row: min_consonants_sequence_length(str(row['Domain Name'])), axis=1)

    df['Vowels Freq'] = df.drop(['Family', 'Label'], axis=1).apply(
        lambda row: vowels_frequency(str(row['Domain Name'])), axis=1)

    df['Max Vowels Seq Length'] = df.drop(['Family', 'Label'], axis=1).apply(
        lambda row: max_vowels_sequence_length(str(row['Domain Name'])), axis=1)

    df['Min Vowels Seq Length'] = df.drop(['Family', 'Label'], axis=1).apply(
        lambda row: min_vowels_sequence_length(str(row['Domain Name'])), axis=1)

    df['Digits Freq'] = df.drop(['Family', 'Label'], axis=1).apply(
        lambda row: digits_frequency(str(row['Domain Name'])), axis=1)

    df['Max Digits Seq Length'] = df.drop(['Family', 'Label'], axis=1).apply(
        lambda row: max_digits_sequence_length(str(row['Domain Name'])), axis=1)

    df['Min Digits Seq Length'] = df.drop(['Family', 'Label'], axis=1).apply(
        lambda row: min_digits_sequence_length(str(row['Domain Name'])), axis=1)

    df['Hyphen Freq'] = df.drop(['Family', 'Label'], axis=1).apply(
        lambda row: hyphen_frequency(str(row['Domain Name'])), axis=1)

    df['Max Hyphen Seq Length'] = df.drop(['Family', 'Label'], axis=1).apply(
        lambda row: max_hyphen_sequence_length(str(row['Domain Name'])), axis=1)

    df['Min Hyphen Seq Length'] = df.drop(['Family', 'Label'], axis=1).apply(
        lambda row: min_hyphen_sequence_length(str(row['Domain Name'])), axis=1)

    df['Max Letter Seq Length'] = df.drop(['Family', 'Label'], axis=1).apply(
        lambda row: max_letter_sequence_length(str(row['Domain Name'])), axis=1)

    df['Min Letter Seq Length'] = df.drop(['Family', 'Label'], axis=1).apply(
        lambda row: min_letter_sequence_length(str(row['Domain Name'])), axis=1)

    return df

def create_high_heterogeneous_dfs(train_df):
    '''
        Create high heterogeneous dataframes for training dataset.

        Args:
            train_df: Train dataframe

        Return:
            Five high heterogeneous dataframes.
    '''

    benign_training_df = train_df[train_df['Label'] == 0]

    # Split the DataFrame into three equal parts
    split_data = np.array_split(benign_training_df, 5)

    # Create three separate DataFrames
    benign_training_df1, benign_training_df2, benign_training_df3, benign_training_df4, benign_training_df5 = split_data

    # Create malicious dataframes.
    malicious_training_df1 = train_df[train_df['Family'].isin(['ramnit', 'kraken'])]

    malicious_training_df2 = train_df[train_df['Family'].isin(['simda''banjori'])]

    malicious_training_df3 = train_df[train_df['Family'].isin(['qakbot', 'cryptolocker'])]

    malicious_training_df4 = train_df[train_df['Family'].isin(['pykspa', 'ramdo'])]

    malicious_training_df5 = train_df[train_df['Family'].isin(['locky', 'corebot', 'dircrypt'])]

    # Create dataframes.
    high_heterogeneous_df1 = create_heterogeneous_df(benign_training_df1, malicious_training_df1)

    high_heterogeneous_df2 = create_heterogeneous_df(benign_training_df2, malicious_training_df2)

    high_heterogeneous_df3 = create_heterogeneous_df(benign_training_df3, malicious_training_df3)

    high_heterogeneous_df4 = create_heterogeneous_df(benign_training_df4, malicious_training_df4)

    high_heterogeneous_df5 = create_heterogeneous_df(benign_training_df5, malicious_training_df5)

    return high_heterogeneous_df1, high_heterogeneous_df2, high_heterogeneous_df3, high_heterogeneous_df4, high_heterogeneous_df5


def split_equally_df(df):
    '''
        Split a dataframe into 5 equal parts.

        Args:
            df: A dataframe

        Return:
            Five different dataframes with equal size
    '''
    # Split the DataFrame into five equal parts
    split_data = np.array_split(df, 5)

    df1, df2, df3, df4, df5 = split_data

    return df1, df2, df3, df4, df5

def create_low_heterogeneous_dfs(train_df):
    '''
        Create low heterogeneous dataframes for training dataset.

        Args:
            train_df: Train dataframe

        Return:
            Five low heterogeneous dataframes.
    '''

    benign_training_df = train_df[train_df['Label'] == 0]

    # Split the DataFrame into three equal parts
    split_data = np.array_split(benign_training_df, 5)

    # Create three separate DataFrames
    benign_training_df1, benign_training_df2, benign_training_df3, benign_training_df4, benign_training_df5 = split_data

    # Create malicious dataframes.
    ramnit_df1, ramnit_df2, ramnit_df3, ramnit_df4, ramnit_df5 = split_equally_df(
        train_df[train_df['Family'] == 'ramnit'])

    kraken_df1, kraken_df2, kraken_df3, kraken_df4, kraken_df5 = split_equally_df(
        train_df[train_df['Family'] == 'kraken'])

    simda_df1, simda_df2, simda_df3, simda_df4, simda_df5 = split_equally_df(train_df[train_df['Family'] == 'simda'])

    banjori_df1, banjori_df2, banjori_df3, banjori_df4, banjori_df5 = split_equally_df(
        train_df[train_df['Family'] == 'banjori'])

    pykspa_df1, pykspa_df2, pykspa_df3, pykspa_df4, pykspa_df5 = split_equally_df(
        train_df[train_df['Family'] == 'pykspa'])

    ramdo_df1, ramdo_df2, ramdo_df3, ramdo_df4, ramdo_df5 = split_equally_df(train_df[train_df['Family'] == 'ramdo'])

    qakbot_df1, qakbot_df2, qakbot_df3, qakbot_df4, qakbot_df5 = split_equally_df(
        train_df[train_df['Family'] == 'qakbot'])

    cryptolocker_df1, cryptolocker_df2, cryptolocker_df3, cryptolocker_df4, cryptolocker_df5 = split_equally_df(
        train_df[train_df['Family'] == 'cryptolocker'])

    locky_df1, locky_df2, locky_df3, locky_df4, locky_df5 = split_equally_df(train_df[train_df['Family'] == 'locky'])

    corebot_df1, corebot_df2, corebot_df3, corebot_df4, corebot_df5 = split_equally_df(
        train_df[train_df['Family'] == 'corebot'])

    dircrypt_df1, dircrypt_df2, dircrypt_df3, dircrypt_df4, dircrypt_df5 = split_equally_df(
        train_df[train_df['Family'] == 'dircrypt'])

    malicious_training_df1 = pd.concat(
        [ramnit_df1, kraken_df1, simda_df1, banjori_df1, pykspa_df1, ramdo_df1, qakbot_df1, cryptolocker_df1, locky_df1,
         corebot_df1, dircrypt_df1], ignore_index=True)

    malicious_training_df2 = pd.concat(
        [ramnit_df2, kraken_df2, simda_df2, banjori_df2, pykspa_df2, ramdo_df2, qakbot_df2, cryptolocker_df2, locky_df2,
         corebot_df2, dircrypt_df2], ignore_index=True)

    malicious_training_df3 = pd.concat(
        [ramnit_df3, kraken_df3, simda_df3, banjori_df3, pykspa_df3, ramdo_df3, qakbot_df3, cryptolocker_df3, locky_df3,
         corebot_df3, dircrypt_df3], ignore_index=True)

    malicious_training_df4 = pd.concat(
        [ramnit_df4, kraken_df4, simda_df4, banjori_df4, pykspa_df4, ramdo_df4, qakbot_df4, cryptolocker_df4, locky_df4,
         corebot_df4, dircrypt_df4], ignore_index=True)

    malicious_training_df5 = pd.concat(
        [ramnit_df5, kraken_df5, simda_df5, banjori_df5, pykspa_df5, ramdo_df5, qakbot_df5, cryptolocker_df5, locky_df5,
         corebot_df5, dircrypt_df5], ignore_index=True)

    # Create low heterogeneous dataframes.
    low_heterogeneous_df1 = create_heterogeneous_df(benign_training_df1, malicious_training_df1)

    low_heterogeneous_df2 = create_heterogeneous_df(benign_training_df2, malicious_training_df2)

    low_heterogeneous_df3 = create_heterogeneous_df(benign_training_df3, malicious_training_df3)

    low_heterogeneous_df4 = create_heterogeneous_df(benign_training_df4, malicious_training_df4)

    low_heterogeneous_df5 = create_heterogeneous_df(benign_training_df5, malicious_training_df5)

    return low_heterogeneous_df1, low_heterogeneous_df2, low_heterogeneous_df3, low_heterogeneous_df4, low_heterogeneous_df5
def create_heterogeneous_df(benign_df, malicious_df):
    '''
        Create a heterogeneous dataframe that comes from the concatination of a benign df and a malicious df.

        Args:
            benign_df: dataframe with legit domain names
            malicious_df: dataframe with malicious domain names
        Return:
            heterogeneous_df: concat dataframe
    '''

    # Concat dataframes into one.
    heterogeneous_df = pd.concat([benign_df, malicious_df], ignore_index=True)

    # Shuffle the DataFrame
    heterogeneous_df = heterogeneous_df.sample(frac=1, random_state=42, ignore_index=True)

    return heterogeneous_df


if __name__ == "__main__":
    # Fix path
    train_dataset_path = r"C:\Users\plust\OneDrive\Desktop\Jupyter Projects\MSc - DSML\Thesis\data\new_final\train_df.csv"
    test_dataset_path = r"C:\Users\plust\OneDrive\Desktop\Jupyter Projects\MSc - DSML\Thesis\data\new_final\test_df.csv"
    data_folder_path = r"C:\Users\plust\OneDrive\Desktop\Jupyter Projects\MSc - DSML\Thesis\data\new_final"

    # Load train and test dataframe.
    train_df = pd.read_csv(filepath_or_buffer=train_dataset_path)
    test_df = pd.read_csv(filepath_or_buffer=test_dataset_path)

    # Compute statistical features.
    final_train_df = feature_extractor(train_df)
    final_test_df = feature_extractor(test_df)

    # Create high heterogeneous dfs.
    high_heterogeneous_df1, high_heterogeneous_df2, high_heterogeneous_df3, high_heterogeneous_df4, high_heterogeneous_df5 = create_high_heterogeneous_dfs(
        train_df)

    # Create λος heterogeneous dfs.
    low_heterogeneous_df1, low_heterogeneous_df2, low_heterogeneous_df3, low_heterogeneous_df4, low_heterogeneous_df5 = create_low_heterogeneous_dfs(
        train_df)

    # Save datasets to a csv file.
    final_train_df.to_csv(path_or_buf=data_folder_path + "/final_train_df.csv", sep=",", index=False, header=True,
                          mode="w")
    final_test_df.to_csv(path_or_buf=data_folder_path + "/final_test_df.csv", sep=",", index=False, header=True,
                         mode="w")

    high_heterogeneous_df1.to_csv(path_or_buf=data_folder_path + "/high_heterogeneous/high_heterogeneous_df1.csv",
                                  sep=",", index=False, header=True, mode="w")
    high_heterogeneous_df2.to_csv(path_or_buf=data_folder_path + "/high_heterogeneous/high_heterogeneous_df2.csv",
                                  sep=",", index=False, header=True, mode="w")
    high_heterogeneous_df3.to_csv(path_or_buf=data_folder_path + "/high_heterogeneous/high_heterogeneous_df3.csv",
                                  sep=",", index=False, header=True, mode="w")
    high_heterogeneous_df4.to_csv(path_or_buf=data_folder_path + "/high_heterogeneous/high_heterogeneous_df4.csv",
                                  sep=",", index=False, header=True, mode="w")
    high_heterogeneous_df5.to_csv(path_or_buf=data_folder_path + "/high_heterogeneous/high_heterogeneous_df5.csv",
                                  sep=",", index=False, header=True, mode="w")

    low_heterogeneous_df1.to_csv(path_or_buf=data_folder_path + "/low_heterogeneous/low_heterogeneous_df1.csv", sep=",",
                                 index=False, header=True, mode="w")
    low_heterogeneous_df2.to_csv(path_or_buf=data_folder_path + "/low_heterogeneous/low_heterogeneous_df2.csv", sep=",",
                                 index=False, header=True, mode="w")
    low_heterogeneous_df3.to_csv(path_or_buf=data_folder_path + "/low_heterogeneous/low_heterogeneous_df3.csv", sep=",",
                                 index=False, header=True, mode="w")
    low_heterogeneous_df4.to_csv(path_or_buf=data_folder_path + "/low_heterogeneous/low_heterogeneous_df4.csv", sep=",",
                                 index=False, header=True, mode="w")
    low_heterogeneous_df5.to_csv(path_or_buf=data_folder_path + "/low_heterogeneous/low_heterogeneous_df5.csv", sep=",",
                                 index=False, header=True, mode="w")