import traintest as tt
import time


def costing_time(start, mid, end):
    print('Training costs: ' + str(mid - start) + 's')
    print('Testing costs: ' + str(end - mid) + 's\n')
    print('Experiment terminates, total time cost: ' + str(end - start) + 's')
    print('=================================================================\n')


def run(file_path, filter_list=None):
    training_dataset, testing_dataset, type_count = tt.generate_dataset(file_path)
    types = list(type_count.keys())
    results, word_left_count = [], []
    # total_testing = len(testing_dataset)
    print('=================================================================')
    print('Baseline experiment is starting...')
    start = time.process_time()
    '''
        word_count: {type_name: {word: word_freq}}
        voc: vocabulary sorting by alphabetically
        voc_count: {word: word_freq} in vocabulary
    '''
    word_count, voc, voc_count = tt.count_word_by_ex(training_dataset, 0)
    baseline_left = len(voc)

    word_prob_t1 = tt.model_building(word_count, voc, types, 0, 0.5 )
    t1_time = time.process_time()

    real_class, pred_class = tt.compute_score(type_count, testing_dataset, word_prob_t1, voc, types, 0)
    baseline_result = tt.result_analysis(real_class, pred_class, types)

    t2_time = time.process_time()
    costing_time(start, t1_time, t2_time)

    for ex in range(1, 3):
        print('=================================================================')
        if ex == 1:
            print('Stopword experiment is starting...')
        if ex == 2:
            print('Wordlength experiment is starting...')

        start = time.process_time()
        word_count, voc, _ = tt.count_word_by_ex(training_dataset, ex)
        word_prob = tt.model_building(word_count, voc, types, ex)
        t1_time = time.process_time()

        real_class, pred_class = tt.compute_score(type_count, testing_dataset, word_prob, voc, types, ex)
        tt.result_analysis(real_class, pred_class, types)
        t2_time = time.process_time()
        costing_time(start, t1_time, t2_time)

    if filter_list:
        print('=================================================================')
        print('Infrequent word filtering experiment is starting...\n')
        # total_freq = tt.sum_total_freq(word_prob_t1)
        voc_length = baseline_left

        for filt in filter_list:
            if filt.isdigit():
                print('Remove the word with frequency = ' + str(filt) + '\n')
            else:
                print('Remove the top ' + str(filt) + ' most frequency words\n')
            rmv_words = tt.remove_words_by_filter(voc_count, filt, voc_length)

            start = time.process_time()
            word_count, voc, _ = tt.count_word_by_ex(training_dataset, 3, rmv_words)
            word_left_count.append(len(voc))

            word_prob = tt.model_building(word_count, voc, types, 3)

            t1_time = time.process_time()

            real_class, pred_class = tt.compute_score(type_count, testing_dataset, word_prob, voc, types, 3)
            result = tt.result_analysis(real_class, pred_class, types)
            results.append(result)

            t2_time = time.process_time()
            costing_time(start, t1_time, t2_time)
    print(baseline_left, baseline_result)
    print(word_left_count)
    print(results)
    tt.stats_plot([baseline_left], word_left_count, [baseline_result], list(results))
    return [baseline_left], word_left_count, list(baseline_result), results


start_time = time.process_time()
print('\n\nThe program is running\n\n')

DATA_FILE_PATH = '../dataset/hns_test.csv'
FILTER_LIST = ['1', '5', '10', '15', '20', '5%', '10%', '15%', '20%', '25%']

X00, X, Y00, Y = run(DATA_FILE_PATH, FILTER_LIST)


end_time = time.process_time()

print('\n\nThe program terminates, total time cost: ' + str(end_time - start_time) + 's')

