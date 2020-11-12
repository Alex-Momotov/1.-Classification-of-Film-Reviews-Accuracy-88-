from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import SGDClassifier
from memory_profiler import memory_usage
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from collections import Counter
from zipfile import ZipFile
from time import time
import numpy as np
import pickle
import re


def memory_time(func):
    """Decorator which prints time and memory taken by the client function."""
    def inner(*args, **kwargs):
        t_before = time()
        mem_before = memory_usage()[0]
        result = func(*args, **kwargs)
        t_after = time()
        mem_after = memory_usage()[0]
        t_elapsed = round(t_after - t_before, 2)
        m_consumed = round(mem_after - mem_before, 2)
        print(f'\"{func.__name__}\" finished, took: {t_elapsed} sec, consumed: {m_consumed} Mb')
        print(f'\tcurrent memory: {memory_usage()[0]} Mb')
        return result
    return inner


@memory_time
def parse_data(dataset='positive', sample_size=10000):
    if dataset == 'positive':
        path = r'data/trn_positives.zip'
    elif dataset == 'negative':
        path = r'data/trn_negatives.zip'
    elif dataset == 'testing':
        path = r'data/testing.zip'
    bodies = []
    titles = []
    with ZipFile(path, 'r') as myzip:
        name_list = myzip.namelist()
        for num, name in enumerate(name_list, start=1):
            with myzip.open(name, 'r') as myzip_data:
                data = pickle.load(myzip_data)
                bodies.append(data['body'])
                titles.append(data['title'])
            if num == sample_size: break
    return bodies, titles


@memory_time
def data_cleaning(data):
    """Punctuation removal, tokenisation, stemming."""
    sno = SnowballStemmer('english')                     # creating english stemmer
    pattern = re.compile(r'[^a-zA-Z\s]')                 # configuring re module to remove all punctuation
    stop_words = set(stopwords.words('english'))         # make a set of stopwords
    clean_data = []
    for text in data:
        step1 = pattern.sub('', text)                            # removes all punctuation
        step2 = step1.lower().split()                            # makes all lowercase and splits
        step3 = [wrd for wrd in step2 if wrd not in stop_words]  # remove stop words
        step4 = [sno.stem(word) for word in step3]               # returns stem of each word
        clean_data.append(step4)
    return clean_data


@memory_time
def create_lexicon(data, ignore_low, ignore_high):
    """Returns dictionary-like lexicon, removes given number of most/least frequently occuring words."""
    data_flat = [word for sublist in data for word in sublist]
    data_counted = Counter(data_flat)
    to_delete = []
    for key, val in data_counted.items():
        if val < ignore_low: to_delete.append(key)
    for key in sorted(data_counted, key=data_counted.get, reverse=True)[:ignore_high]:
        to_delete.append(key)
    for key in to_delete:
        del data_counted[key]
    lexicon = list(data_counted)
    lexicon.sort()
    lexicon_dict = {}
    for idx, word in enumerate(lexicon):
        lexicon_dict[word] = idx
    return lexicon_dict


@memory_time
def create_BOW_matrix(data, lexicon):
    """Creates and saves BOW matrix on disc, returns NumPy memory map of it."""
    list_of_counters = [Counter(text) for text in data]
    BOW_matrix = np.zeros([len(data),len(lexicon)], dtype = np.float16)
    for index, counter_obj in enumerate(list_of_counters):
        for key, val in counter_obj.items():
            try:
                word_idx = lexicon[key]
                BOW_matrix[index,word_idx] = val
            except KeyError: pass
    np.save("Processing stages/BOW_matrix.npy", BOW_matrix)
    BOW_matrix = np.load("Processing stages/BOW_matrix.npy", mmap_mode='r+')
    return BOW_matrix


@memory_time
def create_TFIDF_matrix(BOW_matrix):
    """Converts BOW to TF-IDF matrix saved on disc, returns NumPy memory map of it."""
    def idf(column):
        count = np.count_nonzero(column)
        if count > 0:
            idf_score = np.log(len(column) / count, dtype=np.float32)
        else:
            idf_score = 0
        return idf_score
    idf_scores = np.apply_along_axis(idf, 0, BOW_matrix)
    TFIDF_matrix = idf_scores * BOW_matrix
    np.save("Processing stages/TFIDF_matrix.npy", TFIDF_matrix)
    TFIDF_matrix = np.load("Processing stages/TFIDF_matrix.npy", mmap_mode='r')
    return TFIDF_matrix


@memory_time
def normalise_unit_vec(BOW_matrix):
    """Converts TF-IDF feature vectors into unit vectors, returns NumPy memory map of the new matrix."""
    sq = ((BOW_matrix ** 2).sum(1)) ** 0.5
    TFIDF_matrix_normed = BOW_matrix / sq[:, None]
    np.save("Processing stages/TFIDF_matrix_normed.npy", TFIDF_matrix_normed)
    TFIDF_matrix_normed = np.load("Processing stages/TFIDF_matrix_normed.npy", mmap_mode='r')
    return TFIDF_matrix_normed


#%% Load data
sample_size = 2900
data_pos, titles_pos = parse_data(dataset='positive', sample_size=sample_size)
data_neg, titles_neg = parse_data(dataset='negative', sample_size=sample_size)

#%%  Data cleaning - tokenisation, stemming, stopwords
clean_data_pos = data_cleaning(data_pos)
clean_data_neg = data_cleaning(data_neg)
data_all = clean_data_pos + clean_data_neg
titles_all = titles_pos + titles_neg

positive_labels = [1 for x in range(len(titles_pos))]
negative_labels = [0 for y in range(len(titles_neg))]
target = positive_labels + negative_labels

#%% Create lexicon
lexicon = create_lexicon(data_all, 2, 20)
print('lexicon length:', len(lexicon))

#%% Create Bag of Words (BOW)
BOW_matrix = create_BOW_matrix(data_all, lexicon)
print('BOW matrix shape:', BOW_matrix.shape)
print(type(BOW_matrix))

#%% Convert BOW to TF-IDF matrix
TFIDF_matrix = create_TFIDF_matrix(BOW_matrix)

#%% Normalise feature-vectors within TF-IDF as unit vectors
TFIDF_matrix_normed = normalise_unit_vec(TFIDF_matrix)

#%% Free up some memory
del data_all
del clean_data_pos
del clean_data_neg
del data_pos
del data_neg
del TFIDF_matrix
del BOW_matrix

#%%    ### Classifying with Nearest Centroid and Stochastic Gradient Descent ###

# Nearest Centroid
clf_nc = NearestCentroid(metric='euclidean', shrink_threshold=None)
clf_nc.fit(TFIDF_matrix_normed, target)
centroid_score = cross_val_score(clf_nc, TFIDF_matrix_normed, target, cv=5).mean()
print("Nearest Centroid score:", centroid_score)

#%% Stochastic Gradient Descent
clf_sgd = SGDClassifier(random_state=46, max_iter=45, tol=0.001)
clf_sgd.fit(TFIDF_matrix_normed, target)
sgd_score = cross_val_score(clf_sgd, TFIDF_matrix_normed, target, cv=5).mean()
print("Stochastic Gradient Descent score:", sgd_score)

#%%    ### Analysing fail cases ###

# Delete existing memory-map matrix
del TFIDF_matrix_normed

# Load new data that classifier doesn't know
data_experiment, titles_experiment = (parse_data(dataset='positive', sample_size=3000))
data_experiment = data_experiment[-100:]
titles_experiment = np.array(titles_experiment[-100:])
clean_data_experiment = data_cleaning(data_experiment)

# Make BOW matrix, convert to TF-IDF representation, transform to unit vector form
BOW_matrix_experiment = create_BOW_matrix(clean_data_experiment, lexicon)
TFIDF_experiment = create_TFIDF_matrix(BOW_matrix_experiment)
TFIDF_norm_experiment = normalise_unit_vec(TFIDF_experiment)
print(TFIDF_norm_experiment.shape)

# Predict new data items with SGD classifier
predicted_scores_experiment = clf_sgd.predict(TFIDF_norm_experiment)

# Show file numbers and movie titles of some cases where our classifier failed
mask_scores = predicted_scores_experiment == 0
doc_numbers = [3000 + x for x, y in enumerate(mask_scores, start=1) if y]
failed_case_titles = titles_experiment[mask_scores]
for doc_num, title in zip(doc_numbers, failed_case_titles):
    print(doc_num, title)

#%%    ### Using trained classifier to predict new movies

# Delete existing memory-map matrices
del BOW_matrix_experiment
del TFIDF_experiment
del TFIDF_norm_experiment

# Load and clean new data
data_new, titles_new = parse_data(dataset='testing')
clean_data_new = data_cleaning(data_new)
print(len(clean_data_new))

# Create bag of words (BOW)
BOW_matrix_new = create_BOW_matrix(clean_data_new, lexicon)
print(BOW_matrix_new.shape)

# Convert BOW to TF-IDF matrix
TFIDF_new = create_TFIDF_matrix(BOW_matrix_new)

# Convert TF-IDF to unit vector form
BOW_matrix_idf_norm_new = normalise_unit_vec(TFIDF_new)

# Free up some memory
del data_new
del clean_data_new
del BOW_matrix_new
del TFIDF_new

# Classification with Nearest Centroid
predicted_scores_nc = clf_nc.predict(BOW_matrix_idf_norm_new)

# Classification with Stochastic Gradient Descent
predicted_scores_sgd = clf_sgd.predict(BOW_matrix_idf_norm_new)

# Count the number of good and bad movie predictions with each classifier
print(Counter(predicted_scores_nc))
print(Counter(predicted_scores_sgd))

# Calculate percent matched predictions between two classifiers
matches = 0.
for x, y in zip(predicted_scores_nc, predicted_scores_sgd):
    if x == y: matches += 1
print('Percent matches', (matches / len(predicted_scores_sgd)) * 100)

# Separate new data into two lists of good and bad movies using SGD results
titles_new = np.array(titles_new)
mask_good_movies = predicted_scores_sgd == 1
mask_bad_movies = predicted_scores_sgd == 0
good_movies = titles_new[mask_good_movies]
bad_movies = titles_new[mask_bad_movies]
print(len(good_movies) + len(bad_movies))

