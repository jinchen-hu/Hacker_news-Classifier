import traintest as tt
import time


def run(file_path, ex=0):
    start = time.process_time()

    training_dataset, testing_dataset = tt.generate_dataset(file_path)

    word_count, voc, removed_words = tt.count_word_by_ex(training_dataset, ex)

    word_prob = tt.model_building(word_count, voc, ex)

    t1_time = time.process_time()

    tt.compute_score(training_dataset, testing_dataset, word_prob, voc, ex)

    t2_time = time.process_time()

    print(t1_time-start)
    print(t2_time-t1_time)


DATA_FILE_PATH = '../dataset/hns_2018_2019.csv'
run(DATA_FILE_PATH)

run(DATA_FILE_PATH, 1)
