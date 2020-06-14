import task1 as t1

DATA_FILE_PATH = '../dataset/hns_2018_2019.csv'
training_dataset, testing_dataset = t1.generate_dataset(DATA_FILE_PATH)

word_count, voc, removed_words = t1.count_word(training_dataset)

_, word_prob = t1.model_building(word_count, voc)

# print(word_prob)
# print(t1.compute_type_log(training_dataset))
t1.compute_score(training_dataset, testing_dataset, word_prob, voc)