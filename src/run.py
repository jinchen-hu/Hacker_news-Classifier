import traintest as tt
import time


def costing_time(start, mid, end):
    print('Training costs: ' + str(mid - start) + 's')
    print('Testing costs: ' + str(end - mid) + 's\n')
    print('Experiment terminates, total time cost: ' + str(end - start) + 's')
    print('=================================================================\n')


def run(file_path, filter_list=None):
    training_dataset, testing_dataset = tt.generate_dataset(file_path)
    results, word_left_count = [], []
    # total_testing = len(testing_dataset)
    print('=================================================================')
    print('Baseline experiment is starting...')
    start = time.process_time()
    word_count, voc,_, types = tt.count_word_by_ex(training_dataset, 0)
    baseline_left = len(voc)

    word_prob_t1 = tt.model_building(word_count, voc, 0, 0.5, types)
    t1_time = time.process_time()

    real_class, pred_class = tt.compute_score(training_dataset, testing_dataset, word_prob_t1, voc, 0, types)
    baseline_result = tt.result_analysis(real_class, pred_class)

    t2_time = time.process_time()
    costing_time(start, t1_time, t2_time)

    for ex in range(1, 3):
        print('=================================================================')
        if ex == 1:
            print('Stopword experiment is starting...')
        if ex == 2:
            print('Wordlength experiment is starting...')

        start = time.process_time()
        word_count, voc, _, types = tt.count_word_by_ex(training_dataset, ex)
        word_prob = tt.model_building(word_count, voc, ex, types=types)
        t1_time = time.process_time()

        tt.compute_score(training_dataset, testing_dataset, word_prob, voc, ex, types)

        t2_time = time.process_time()
        costing_time(start, t1_time, t2_time)

    if filter_list:
        print('=================================================================')
        print('Infrequent word filtering experiment is starting...\n')
        total_freq = tt.sum_total_freq(word_prob_t1)
        voc_length = baseline_left

        for filt in filter_list:
            if filt.isdigit():
                print('Remove the word with frequency = ' + str(filt) + '\n')
            else:
                print('Remove the top ' + str(filt) + ' most frequency words\n')
            rmv_words = tt.remove_words_by_filter(total_freq, filt, voc_length)

            start = time.process_time()
            word_count, voc, _, types = tt.count_word_by_ex(training_dataset, 3, rmv_words)
            word_left_count.append(len(voc))

            word_prob = tt.model_building(word_count, voc, 3, types=types)

            t1_time = time.process_time()

            real_class, pred_class = tt.compute_score(training_dataset, testing_dataset, word_prob, voc, 3, types)
            result = tt.result_analysis(real_class, pred_class)
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

DATA_FILE_PATH = '../dataset/test.csv'
FILTER_LIST = ['1', '5', '10', '15', '20', '5%', '10%', '15%', '20%', '25%']

X00, X, Y00, Y = run(DATA_FILE_PATH, FILTER_LIST)


end_time = time.process_time()

print('\n\nThe program terminates, total time cost: ' + str(end_time - start_time) + 's')

# print("Remaining words in Vocab: \n", new_vocab_len)
# print("Accuracy: \n", accuracy_diffsmooth)
# plt.style.use('ggplot')
# plt.plot(new_vocab_len, accuracy_diffsmooth, linestyle='--', marker='o', color='b')
# plt.xlabel('Number of words remaining in vocab')
# plt.ylabel('Accuracy')
# plt.title("EXPERIMENT 4 RESULTS")
# plt.show()


