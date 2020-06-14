import traintest as tt
import time


def run(file_path, ex=0):
    print('=================================================================')
    if ex == 0:
        print('Baseline experiment is starting')
    if ex == 1:
        print('Stopword experiment is starting')
    if ex == 2:
        print('Wordlength experiment is starting')

    start = time.process_time()

    training_dataset, testing_dataset = tt.generate_dataset(file_path)

    word_count, voc, removed_words = tt.count_word_by_ex(training_dataset, ex)

    word_prob = tt.model_building(word_count, voc, ex)

    t1_time = time.process_time()

    tt.compute_score(training_dataset, testing_dataset, word_prob, voc, ex)

    t2_time = time.process_time()

    print('Training costs: ' + str(t1_time-start) + 's')
    print('Testing costs: ' + str(t2_time-t1_time) + 's\n')
    print('Experiment terminates, total time cost: ' + str(t2_time - start) + 's')
    print('=================================================================\n')


start_time = time.process_time()
print('\n\nThe program is running\n\n')

DATA_FILE_PATH = '../dataset/hns_2018_2019.csv'

run(DATA_FILE_PATH)

run(DATA_FILE_PATH, 1)

run(DATA_FILE_PATH, 2)

end_time = time.process_time()

print('\n\nThe program terminates, total time cost: ' + str(end_time - start_time) + 's')