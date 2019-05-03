from lib.data_utils import *
from lib.file_utils import *
from lib.time_utils import *
from lib.nlp_util import *
from keras.utils import np_utils
import numpy
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline


"""
HEADER = ['cookie', 'result_name', 'result_provider', 'timestamp', 'query_raw', 'query_string', 'query_normalized',
               'query_language', 'name_chosen', 'result_lat', 'result_lon', 'result_index', 'result_relevance',
               'result_rank', 'query_lat', 'query_lon', 'query_viewport', 'description_str']
"""

#----------------------------------------------------
#  data (Chicago 2017 query and result (POI, RECO)
#----------------------------------------------------
HEADER = ['cookie', 'ppid', 'request_type', 'timestamp', 'query', 'language', 'category', 'place_name', 'query_lat', 'query_lon', 'unknown_0', 'unknown_1', 'place_lat', 'place_lon', 'viewport', 'comment', 'unknown_2', 'source']
DATA = "C:/Users/shong/Documents/data/data_for_query_multiclass_classifier/chicago1year.tsv"

#data = read_file_without_header(DATA)
data = read_tsv_file(DATA, HEADER)


#-----------------
# cleaning data
#-----------------
#print("[log] shape :", data.shape)    # (189576, 18)
#print(data.head(10))
#print(data.cookie)
data['user_id'] = column_factorize(data.cookie)
#print(data.user_id)

data['place_id'] = column_factorize(data.ppid)
#print(data.place_id)

data['time'] = data.apply(lambda row: convertToReadableDate(row.timestamp), axis=1)
#print(data.time)

# sort data with time
time_sorted_data = sort_dataframe_with_column(data, 'time')
#print(time_sorted_data)


#-------------------------
# data frame with query
#-------------------------
column_names = ['user_id', 'place_id', 'place_name', 'time', 'query']
query_place = ['query', 'place_name']
query_data = time_sorted_data[column_names]


#------------------------
# query contains 'near'
#------------------------
QUERY_COL = 'query'
QUERY_STR = 'nearby'
queries_contains_str = get_all_queries_contains(QUERY_STR, query_data, QUERY_COL)
#print(queries_contains_str.unique())
#print(len(queries_contains_str))


#------------------------------------------------
# training data
# label 0 - neural
# label 1 - proximity
# label 2 - popularity
# label 3 - both proximity and popularity
#------------------------------------------------

# initialize the label with 0 (neutral)
#print(query_data[query_place])
query_data['label'] = 0  # default label
#print(type(query_data['query'].loc[0]))  # str
query_data['label'] = query_data['query'].apply(lambda x: assign_label(x))
count_different_label(query_data, 'label')

#  if query is one word, use word2vec, if query has more than one word, doc2vec
import gensim.models as g
pre_trained_doc2vec_model = 'C:/Users/shong/Downloads/enwiki_dbow/enwiki_dbow/doc2vec.bin'

w2v_model = load_google_w2v_model()
d2v_model = g.Doc2Vec.load(pre_trained_doc2vec_model)

query_data['query_vector'] = query_data.apply(lambda row: convert_query_to_vector(row.query, w2v_model, d2v_model), axis=1)
training_cols = ['query_vector', 'label']
classification_data = query_data[training_cols]
print(classification_data)


#------------------------
# prepare training_data
#------------------------
seed = 7
numpy.random.seed(seed)
#dataset = classification_data  # ToDo : should I prepare?
dataset = classification_data.values
X = dataset[0:30,0]
ORG_X = X
print("X[0] : ", X[0])
print("X[0].shape : ", X[0].shape)  # (300, )
print("len(X[0]) : ", len(X[0]))  # length is 300 (array element numbers 300)
X = np.stack(X)
print("X.shape : ", X.shape)

#X = dataset[:,0:301].astype(float)
Y = dataset[0:30,1]
print(dataset.shape)
print("X : ")
print(X)
print("X.shape : ")
print(X.shape)  # (189576,) -  x-dimension with size of 189576 rows
print("Y : ")
print(Y)  # [0 0 2 ... 0 0 0]
print("Y.shape : ")
print(Y.shape)  # (189576,)

# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(query_data['label'])  # e.g. [0. 0. 1.]
print(dummy_y)

# define baseline model
def baseline_model():
    # create model
    model = Sequential()
    #model.add(Dense(8, input_dim=4, activation='relu'))  #  ValueError: Error when checking input: expected dense_1_input to have shape (4,) but got array with shape (1,)
    model.add(Dense(8, input_dim=300, activation='relu'))  # to fix ValueError
    model.add(Dense(3, activation='softmax'))
    # compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def experiment_1():
    estimator = KerasClassifier(build_fn=baseline_model, epochs=20, batch_size=5, verbose=0)
    kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
    results = cross_val_score(estimator, X, dummy_y[0:30], cv=kfold)
    print("Baseline: %.2f%% (%.2f%%)" % (results.mean() * 100, results.std() * 100))

    # prediction test
    print("test : ")
    print(X.shape)
    print(X[0].shape)
    print(baseline_model().predict(X[0:1]))


experiment_1()
