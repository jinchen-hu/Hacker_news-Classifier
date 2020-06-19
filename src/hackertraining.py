
import nltk
from nltk.corpus import stopwords as sw
from nltk.stem import WordNetLemmatizer as wnl
from nltk.tokenize import word_tokenize as wt
from nltk.tokenize import RegexpTokenizer as rt
import numpy as np
import os
import string
import math
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support as prf
import pandas as pd
import re


# Create the folder if it's not existing
def create_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path)


def write_file(folder_path, file_path, data):
    create_dir(folder_path)
    file_path = folder_path + "/" + file_path
    fp = open(file_path, 'w+')
    for i in range(len(data)):
        fp.write(data[i] + '\n')
    fp.close()
    print(file_path + " has been created successfylly\n")


def generate_dataset(data_file_path):
    # Read the data as DataFrame type, use 'Created At' as row labels/indices,
    # extract columns 'Title', 'Post Type', and 'Created At',
    data = pd.read_csv(data_file_path, usecols=['Title', 'Post Type', 'Created At'])
    # Extract data in 2018 as Training data
    training_data = data.loc[data['Created At'] < '2019']
    testing_data = data.loc[data['Created At'] >= '2019']
    classes, class_freq = np.unique(data['Post Type'], return_counts=True)
    print(training_data.shape)
    print(testing_data.shape)
    print(classes)
    return training_data, testing_data, classes


def count_word_by_ex(dataset, ex=0, stop_words=None):
    # Define a dictionary <typename, <word, count>>
    word_count = {}
    voc = set()
    rmv_word = set()
    types = set()
    # Traverse each title
    for index, row in dataset.iterrows():
        type_name = row['Post Type']
        types.add(type_name)
        title_tokens = []
        if ex == 0:
            title_tokens = generate_tokens(row['Title'], rmv_word)
        if ex == 1:
            stop_words = read_stop_word('../dataset/Stopwords.txt')
            title_tokens = generate_tokens_by_stop_words(row['Title'], stop_words)
        if ex == 2:
            title_tokens = generate_tokens_by_wordlength(row['Title'])
        if ex == 3:
            title_tokens = generate_tokens_by_stop_words(row['Title'], stop_words)
        # Traverse the title
        for word in title_tokens:
            # Add the word in vocabulary set
            voc.add(word)
            # If the type name is one of keys
            if type_name in word_count.keys():
                # Check if the word is one of keys in current type
                # Yes -> increment; no -> create a new key/value pair
                if word in word_count[type_name].keys():
                    word_count[type_name][word] += 1
                else:
                    word_count[type_name][word] = 1
            # If the type name is not one key
            # Create a key/value pair
            else:
                word_count[type_name] = {}
                word_count[type_name][word] = 1
    # If the type doesn't contains some words in vocabulary, set its number to 0
    for word in voc:
        for type_name in word_count.keys():
            if word not in word_count[type_name].keys():
                word_count[type_name][word] = 0
    if ex == 0:
        output_path = '../task1'
        write_file(output_path, 'vocabulary.txt', sorted(voc))
        write_file(output_path, 'removed_word.txt', sorted(rmv_word))
    if ex == 1:
        output_path = '../task3/ex1'
        write_file(output_path, 'stopword-vocabulary.txt', sorted(voc))
    if ex == 2:
        output_path = '../task3/ex2'
        write_file(output_path, 'wordlength-vocabulary.txt', sorted(voc))
    return word_count, sorted(voc), sorted(rmv_word), list(types)


def trim_lower_title(title):
    title = title.replace("’", "")
    title = title.replace("'", "")
    title = title.lower()
    return nltk.word_tokenize(title)


def generate_tokens(title, rmv_words):
    # Replace curly quote to vertical quote
    original_tokens = trim_lower_title(title)
    tokens = []
    for word in original_tokens:
        word = word.strip()
        # Remove all punctuations and single letters
        if re.search(r'[\w]+', word) and len(word) > 1:
            for pct in string.punctuation:
                word = word.strip(pct)
            tokens.append(word)
        else:
            rmv_words.add(word)
    return tokens


def generate_baseline_tokens(title):
    remove_words = []




generate_dataset('../dataset/hns_2018_2019.csv')
