import nltk
from nltk import word_tokenize, pos_tag, ne_chunk
import gensim
from gensim.models import Word2Vec
from gensim.models import Phrases

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')


# Name-entity recognition using NLTK
def name_entity_rec(input_str):
    return ne_chunk(pos_tag(word_tokenize(input_str)))


# print part-of-tagging and name-entity recognition
def print_pos_tag_and_ner(input_str):
    str_tokens = word_tokenize(input_str)  # ['stores', 'near', 'me']
    pos_tagged_token_strs = pos_tag(str_tokens)  # [('stores', 'NNS'), ('near', 'IN'), ('me', 'PRP')]
    print(str_tokens)
    print(pos_tagged_token_strs)
    print(ne_chunk(pos_tagged_token_strs))


def example_word_vector_representation():
    from gensim.test.utils import common_texts
    bigram_transformer = Phrases(common_texts)
    model = Word2Vec(bigram_transformer[common_texts], min_count=1)

    print(common_texts)
    print(model)
    vector1 = model['computer']
    vector2 = model['system']
    print(vector1)
    print(vector2)
    sim_val = model.similarity('computer', 'system')
    print(sim_val)  # similarity


# doc2vec
def test_load_pre_trained_doc2vec():
    import gensim.models as g
    pre_trained_model = 'C:/Users/shong/Downloads/enwiki_dbow/enwiki_dbow/doc2vec.bin'
    # load model
    m = g.Doc2Vec.load(pre_trained_model)
    input_str3 = "nearest gas station"
    input_str7 = "mexican restaurants near my location"
    tokens = input_str7.split()
    #print(tokens)
    dv = m.infer_vector(tokens)
    #print(dv)
    #print(len(dv))  # len is 300



def doc2vec_to_wikipedia_article():
    from gensim.corpora.wikicorpus import WikiCorpus
    from gensim.models.doc2vec import Doc2Vec, TaggedDocument
    from pprint import pprint
    import multiprocessing


def is_contain_proximity_substring(org_string, substr):
    ret = 0
    if substr in org_string:
        print(org_string)
        ret = 1
        return ret
    else:
        return ret


def is_contain_popularity_substring(org_string, substr):
    ret = 0
    if substr in org_string:
        print(org_string)
        ret = 2
        return ret
    else:
        return ret


def assign_label(org_string):
    ret = 0
    if "nearby" in org_string:
        print(org_string)
        ret = 1
    elif "near me" in org_string:
        print(org_string)
        ret = 1
    elif "nearest" in org_string:
        print(org_string)
        ret = 1
    elif "best" in org_string:
        print(org_string)
        ret = 2
    elif "nice" in org_string:
        print(org_string)
        ret = 2
    elif "good" in org_string:
        print(org_string)
        ret = 2

    return ret

"""
# example strings
input_str1: str = "stores near me"
input_str2 = "library near me"
input_str3 = "nearest gas station"
input_str4 = "Nearest Airport"
input_str5 = "african restaurant near me"
input_str6 = "mosque near me chicago"
input_str7 = "mexican restaurants near my location"
input_str8 = "best restaurant near me"
input_str9 = "lodging near arlington heights, IL"

# tokenize
print(ne_chunk(pos_tag(word_tokenize(input_str1))))
print(ne_chunk(pos_tag(word_tokenize(input_str2))))
print(ne_chunk(pos_tag(word_tokenize(input_str3))))
print(ne_chunk(pos_tag(word_tokenize(input_str4))))
print(ne_chunk(pos_tag(word_tokenize(input_str5))))
print(ne_chunk(pos_tag(word_tokenize(input_str6))))
print(ne_chunk(pos_tag(word_tokenize(input_str7))))

query_tokens = word_tokenize(input_str9)  # ['stores', 'near', 'me']

pos_tagged = pos_tag(query_tokens)  # [('stores', 'NNS'), ('near', 'IN'), ('me', 'PRP')]

print(ne_chunk(pos_tagged))
"""


"""
# word2vec representation
import gensim.models
model = "C:/Users/shong/Documents/english_wikipedia_trained_model/model.txt"
word_vectors = gensim.models.KeyedVectors.load_word2vec_format(model, binary=False)
#print(word_vectors.most_similar("vacation_NOUN"))
#print(word_vectors.most_similar(positive=['woman_NOUN', 'king_NOUN'], negative=['man_NOUN']))


print(word_vectors.most_similar("restaurant_NOUN")) # do not work 

"""

""" take too much time 
# using fast text pre-trained model
import gensim.models

import io

def load_vectors(file_name):
    fin = io.open(file_name, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = map(float, tokens[1:])
    return data


fasttext_model_vector = "C:/Users/shong/Downloads/cc.en.300.vec"
vectors = load_vectors(fasttext_model_vector)
print(vectors["restaurant"])
"""


#  using glove (Global Vectors for Word Representation)
""" error with Corpus 
import itertools
from gensim.models.word2vec import Text8Corpus
from glove import Corpus, Glove
sentences = list(itertools.islice(Text8Corpus('text8'),None))
corpus = Corpus()
corpus.fit(sentences, window=10)
glove = Glove(no_components=100, learning_rate=0.05)
glove.fit(corpus.matrix, epochs=30, no_threads=4, verbose=True)
"""

# GloVe
"""
from gensim.scripts.glove2word2vec import glove2word2vec
glove_input_file = 'glove.6B.100d.txt'
word2vec_output_file = 'glove.6B.100d.txt.word2vec'
glove2word2vec(glove_input_file, word2vec_output_file)
"""
