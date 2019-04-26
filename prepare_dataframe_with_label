from lib.data_utils import *
from lib.file_utils import *
from lib.time_utils import *
from lib.nlp_util import *
from keras.utils import np_utils

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
query_data['label'] = 0  # default
print(type(query_data['query'].loc[0]))  # str
"""
query_data['label'] = query_data['query'].apply(lambda x: is_contain_proximity_substring(x, "near me"))
query_data['label'] = query_data['query'].apply(lambda x: is_contain_proximity_substring(x, "nearby"))
query_data['label'] = query_data['query'].apply(lambda x: is_contain_popularity_substring(x, "best"))
"""
query_data['label'] = query_data['query'].apply(lambda x: assign_label(x))

count_different_label(query_data, 'label')


# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(query_data['label'])  # e.g. [0. 0. 1.]
print(dummy_y)


# TODO : label 3 with both proximity and popularity
# TODO : convert query to vector, with doc2vec, we train the model
#print(query_data)
