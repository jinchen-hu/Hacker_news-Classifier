import pandas as pds
import nltk
import string
nltk.download('punkt')
# Definie the path of data set
DATA_FILE_PATH = '../dataset/hns_2018_2019.csv'
# Read the data as DataFrame type, use 'Created At' as row labels/indices,
# extract columns 'Title', 'Post Type', and 'Created At',
data = pds.read_csv(DATA_FILE_PATH,
                    index_col='Created At',
                    usecols=['Title', 'Post Type', 'Created At'])

# Extract data in 2018 as Trainning data
training_data = data.loc[data.index < '2019']
testing_data = data.loc[data.index >= '2019']
print(training_data.shape)
print(testing_data.shape)
# A set storing removed words
removed_words = set()
# A set storing vocabularies
voc = set()
# A map<key=type, value = count>
type_count = {}

for index, row in training_data.iterrows():
    type_name = row['Post Type']
    if type_name in type_count.keys():
        type_count[type_name] += 1
    else:
        type_count[type_name] = 1


print(type_count)



