import hackertraining as hk


DATA_FILE_PATH = '../dataset/hns_2018_2019.csv'
'''
    training_data: training data extracted from csv
    testing_data: testing data extracted from csv
    type_count: a dic, {type_name: type_freq}
'''
training_data, testing_data, type_count = hk.generate_dataset(DATA_FILE_PATH)
types = type_count.keys()

'''
    word_count: {type_name: {word: word_freq}}
    voc: vocabulary sorting by alphabetically
    voc_count: {word: word_freq} in vocabulary
'''
word_count, voc, voc_count = hk.count_word_by_ex(training_data, 0)
print(word_count)