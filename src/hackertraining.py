
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import numpy as np
import os
import string
import math
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support
import pandas as pd
import re


# Create the folder if it's not existing
def create_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path)


def write_file(folder_path, file_path, data):
    create_dir(folder_path)
    file_path = folder_path + "/" + file_path
    fp = open(file_path, 'w+', encoding='utf-8')
    for i in range(len(data)):
        fp.write(data[i] + '\n')
    fp.close()
    print(file_path + " has been created successfylly\n")


def write_model_two(words_prob, types, out_path, file_path, file_prettier_path):
    create_dir(out_path)
    file_path = out_path + file_path
    file_prettier_path = out_path + file_prettier_path
    fp = open(file_path, 'w+', encoding='utf-8')
    fp_prettier = open(file_prettier_path, 'w+', encoding='utf-8')
    line_counter = 1
    # words_prob = sorted(words_prob)
    for word in sorted(words_prob):
        # Write the line counter
        fp.write(str(line_counter) + "  ")
        fp_prettier.write(str(line_counter) + "  ")
        # Write the word
        fp.write(word + "  ")
        fp_prettier.write(word + "  ")
        # Write the freq and prob in each type
        stat = words_prob[word]
        for i in range(len(stat)):
            fp.write(str(stat[i][0]) + '  ' + str(stat[i][1]) + '  ')
            fp_prettier.write(types[i] + ": " + str(stat[i]) + '  ')
        fp.write('\n')
        fp_prettier.write('\n')
        line_counter += 1

    fp.close()
    fp_prettier.close()
    print(file_path + ' has been created successfully\n')
    print(file_prettier_path + ' has been created successfully\n')


def write_model(words_prob, out_path, file_path):
    create_dir(out_path)
    file_path = out_path + file_path
    fp = open(file_path, 'w+', encoding='utf-8')
    line_counter = 1
    # words_prob = sorted(words_prob)
    for word in sorted(words_prob):
        # Write the line counter
        fp.write(str(line_counter) + "  ")
        # Write the word
        fp.write(word + "  ")
        # Write the freq and prob in each type
        stat = words_prob[word]
        for i in range(len(stat)):
            fp.write(str(stat[i][0]) + '  ' + str(stat[i][1]) + '  ')
        fp.write('\n')
        line_counter += 1

    fp.close()
    print(file_path + ' has been created successfully\n')


def read_stop_word(file_path):
    fp = open(file_path, 'r+', encoding='utf-8')
    stop_word = fp.read().replace("'", "")
    stop_word = stop_word.replace("’", "")
    stop_word.lower()
    stop_word.splitlines()
    fp.close()
    return stop_word


def generate_dataset(data_file_path):
    # Read the data as DataFrame type, use 'Created At' as row labels/indices,
    # extract columns 'Title', 'Post Type', and 'Created At',
    data = pd.read_csv(data_file_path, usecols=['Title', 'Post Type', 'Created At'])
    # Extract data in 2018 as Training data
    training_data = data.loc[data['Created At'] < '2019']
    testing_data = data.loc[data['Created At'] >= '2019']
    print(training_data.shape)
    print(testing_data.shape)
    type_count = count_type(training_data)
    return training_data, testing_data, type_count


def count_type(training_data):
    types, type_freq = np.unique(training_data['Post Type'], return_counts=True)
    type_count = {}
    sorted_dic = {}
    for i in range(len(types)):
        type_count[types[i]] = type_freq[i]
    if 'story' in type_count.keys():
        sorted_dic['story'] = type_count['story']
    if 'ask_hn' in type_count.keys():
        sorted_dic['ask_hn'] = type_count['ask_hn']
    if 'show_hn' in type_count.keys():
        sorted_dic['show_hn'] = type_count['show_hn']
    if 'poll' in type_count.keys():
        sorted_dic['poll'] = type_count['poll']
    if len(type_count.keys()) > 4:
        return type_count
    print(sorted_dic)
    return sorted_dic


def count_word_by_ex(dataset, ex=0, stop_words=None):
    # Define a dictionary <typename, <word, count>>
    word_count = {}
    voc = []
    rmv_word = []
    # Traverse each title
    for index, row in dataset.iterrows():
        type_name = row['Post Type']
        title_tokens = []
        if ex == 0:
            title_tokens = generate_baseline_tokens(row['Title'], rmv_word)
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
            voc.append(word)
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
    voc, voc_freq = np.unique(voc, return_counts=True)
    voc_count = {}
    for word in voc:
        for type_name in word_count.keys():
            if word not in word_count[type_name].keys():
                word_count[type_name][word] = 0
    if ex == 0:
        output_path = '../task1'
        write_file(output_path, 'vocabulary.txt', voc)
        write_file(output_path, 'removed_word.txt', np.unique(rmv_word))
        for i in range(len(voc)):
            voc_count[voc[i]] = voc_freq[i]
    if ex == 1:
        output_path = '../task3/ex1'
        write_file(output_path, 'stopword-vocabulary.txt', voc)
    if ex == 2:
        output_path = '../task3/ex2'
        write_file(output_path, 'wordlength-vocabulary.txt', voc)
    return word_count, voc, voc_count


def trim_lower_title(title):
    title = title.replace("’", "")
    title = title.replace("'", "")
    title = title.replace(".", "")
    title = title.replace("/", " ")
    title = title.lower()
    return nltk.word_tokenize(title)


def generate_baseline_tokens(title, rmv_words):
    # Replace curly quote to vertical quote
    original_tokens = trim_lower_title(title)
    tokens = []
    for word in original_tokens:
        word = word.strip()
        # Remove all punctuations and single letters
        if re.search(r'[\w]+', word) and len(word) > 1 and not word[0].isdecimal() and word not in stopwords.words('english'):
            for pct in string.punctuation:
                word = word.strip(pct)
            tokens.append(word)
        else:
            rmv_words.append(word)
    return tokens


def generate_tokens_by_stop_words(title, stop_words):
    # Check if the title contains any stop word, if does, remove it
    original_tokens = trim_lower_title(title)
    tokens = []
    for word in original_tokens:
        if word not in stop_words and len(word) > 1 and re.search(r'[\w]+', word) and word not in stopwords.words('english'):
            # Remove the punctuations form two sides
            for pct in string.punctuation:
                word = word.strip(pct)
            tokens.append(word)
    return tokens


def generate_tokens_by_wordlength(title):
    # Replace curly quote to vertical quote
    original_tokens = trim_lower_title(title)
    tokens = []
    for word in original_tokens:
        word = word.strip()
        # Remove all punctuations and single letters
        if re.search(r'[\w]+', word) and 2 <= len(word) <= 9 and word not in stopwords.words('english'):
            for pct in string.punctuation:
                word = word.strip(pct)
            tokens.append(word)
    return tokens


def remove_words_by_filter(total_freq, filter_factor, word_prob_length):
    removed_words = []
    if filter_factor.isdigit():
        filter_facotr = int(filter_factor)
        for word, freq in total_freq.items():
            if freq <= filter_facotr:
                removed_words.append(word)
    else:
        filter_factor = int(filter_factor[:-1])
        # Sort the total_freq by the orter of descending
        total_freq_list = sorted(total_freq.items(), key=lambda kv: kv[1], reverse=True)
        # Get the frequency last word that needs to be removed
        target_num = total_freq_list[int(filter_factor / 100 * word_prob_length)][1]
        for word, freq in total_freq.items():
            if freq >= target_num:
                removed_words.append(word)
    return removed_words

''''
def sum_total_freq(word_prob):
    total_freq = {}
    for word, freq_prob in word_prob.items():
        # Extract the first row
        ls = list(np.array(freq_prob)[:, 0])
        # Sum the frequency
        total_freq[word] = sum([int(i) for i in ls])
    return total_freq
'''


# P(word|type) = (freq_in_type + smo)/(total_words_in_type + voc*smo)
def model_building(word_count, voc, types, ex=0, smooth_factor=0.5):
    voc_size = len(voc)
    total_in_type = total_words_types(word_count)
    for type_name, counts in word_count.items():
        for word, frequency in counts.items():
            # Compute the probability with smooth factor
            prob = ("%.9f" % ((counts[word] + smooth_factor) / (total_in_type[type_name] + smooth_factor * voc_size)))
            # Change to a list
            counts[word] = []
            # Append frequency and probability to the list
            counts[word].append(frequency)
            counts[word].append(prob)

    words_prob = {}
    # Traverse the vocabulary
    for word in voc:
        # A list storing the freq and prob of each word in different types
        words_prob[word] = []
        for i in range(len(types)):
            # Only count the existing type
            if types[i] in word_count.keys():
                # Append the list of freq and prob to the list
                words_prob[word].append(word_count[types[i]][word])
    if ex == 0:
        write_model_two(words_prob, types, out_path='../task1', file_path='/model-2018.txt', file_prettier_path='/model-prettier.txt')
        write_model(words_prob, out_path='../task1', file_path='/model-2018.txt')
    if ex == 1:
        write_model_two(words_prob, types, out_path='../task3/ex1', file_path='/stopword-model.txt', file_prettier_path='/stopword-model-prettier.txt')
        write_model(words_prob, out_path='../task3/ex1', file_path='/stopword-model.txt')
    if ex == 2:
        write_model_two(words_prob, types, out_path='../task3/ex2', file_path='/wordlength-model.txt', file_prettier_path='/wordlength-model-prettier.txt')
        write_model(words_prob, out_path='../task3/ex2', file_path='/wordlength-model.txt')
    return words_prob



def compute_type_log(dataset):
    type_count = {}
    for index, row in dataset.iterrows():
        type_name = row['Post Type']
        # Count the number of each type
        if type_name in type_count.keys():
            type_count[type_name] += 1
        else:
            type_count[type_name] = 1
    for tp, num in type_count.items():
        type_count[tp] = math.log10(num/len(dataset))
    return type_count



# Compute the total words in each type
def total_words_types(word_count):
    total_words = {}
    for key, value in word_count.items():
        total_words[key] = 0
        for word, count in value.items():
            total_words[key] += count
    return total_words



# Score(title|type)=log(P(tp|types)) + sum(log(word|tp))
def compute_score(training_data, testing_data, words_prob, voc, types, ex=0):
    # Get the prob of each type
    type_prob = compute_type_log(training_data)
    # Compute log value of each word in each type
    log_values = {}
    lent = len(words_prob[list(words_prob.keys())[0]])
    for word in voc:
        # Store <type, log_value>
        log_values[word] = {}
        for i in range(lent):
            log_values[word][types[i]] = math.log10(float(words_prob[word][i][1]))
    file_path, fp = '', ''
    if ex == 0:
        folder_path = '../task2'
        create_dir(folder_path)
        file_path = folder_path + '/baseline-result.txt'
    if ex == 1:
        folder_path = '../task3/ex1'
        create_dir(folder_path)
        file_path = folder_path + '/stopword-result.txt'
    if ex == 2:
        folder_path = '../task3/ex2'
        create_dir(folder_path)
        file_path = folder_path + '/wordlength-result.txt'
    if ex != 3:
        fp = open(file_path, 'w+', encoding='utf-8')
    line_counter = 0
    real_class, pred_class = [],[]
    # Manipulate testing dataset
    for index, row in testing_data.iterrows():
        line_counter += 1
        # Lowercase the title
        original_title = row['Title']
        title = original_title.lower()
        real_type = row['Post Type']
        # A list storing scores of each type
        scores = []
        for i in range(lent):
            scores.append(type_prob[types[i]])
        # Tokenize the tile
        original_tokens = nltk.word_tokenize(title)
        tokens = []

        # Generate new tokens confirmed to the rules of training data set
        for word in original_tokens:
            word.strip()
            if re.search(r'[\w]+', word) and len(word) > 1:
                for pct in string.punctuation:
                    word = word.strip(pct)
                tokens.append(word)

        # Iterate the tokens, and compute the log value
        for word in tokens:
            # Only compute the word in the vocabulary
            if word in voc:
                for i in range(lent):
                    scores[i] += log_values[word][types[i]]

        classifier_type = types[scores.index(max(scores))]
        if ex != 3:
            fp.write(str(line_counter) + '  ')
            fp.write(original_title + '  ')
            fp.write(classifier_type + '  ')
            for i in range(lent):
                fp.write(str(scores[i]) + '  ')
            fp.write(real_type + '  ')
            fp.write(str(classifier_type == real_type) + '\n')

        real_class.append(real_type)
        pred_class.append(classifier_type)
    print(file_path + ' has been created successfully')
    return real_class, pred_class