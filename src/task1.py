import pandas as pds
import nltk
nltk.download('punkt')
# Definie the path of data set
DATA_FILE_PATH = '../dataset/hns_2018_2019.csv'
# Read the data as DataFrame type, use 'Created At' as row labels/indices,
# extract columns 'Title', 'Post Type', and 'Created At',
# set column 'Created At' as date times
data = pds.read_csv(DATA_FILE_PATH,
                    index_col='Created At',
                    usecols=['Title', 'Post Type', 'Created At'])

# Extract data in 2018 as Trainning data
training_data = data.loc[data.index < '2019']
test_data = data.loc[data.index >= '2019']
print(training_data.shape)
print(test_data.shape)

temp = data.iloc[2]['Title']
print(temp)

print(nltk.word_tokenize(temp))




