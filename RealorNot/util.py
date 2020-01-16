
import collections
import os


def load_data_2_embed(train_x, test_x):
    # build the complete vocabulary, then convert text data to list of integers
    word_to_id = build_vocab(train_x)
    train_data_x = file_to_word_ids(train_x, word_to_id)
    test_data_x = file_to_word_ids(test_x, word_to_id)
    vocabulary = len(word_to_id)
    reversed_dictionary = dict(zip(word_to_id.values(), word_to_id.keys()))

    print("\n\n")
#     print(train_data)
#     print(word_to_id)
    print(vocabulary)
    print(" ".join([reversed_dictionary[x] for x in train_data_x[0]]))
    return train_data_x, test_data_x, vocabulary, reversed_dictionary

def build_vocab(data):
    data = get_train_words_set(data)

    counter = collections.Counter(data)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

    words, _ = list(zip(*count_pairs))
    word_to_id = dict(zip(words, range(len(words))))

    return word_to_id

def get_train_words_set(data):
    result = [char for string in data for char in string.split()]
    #print(result)
    return result

def read_words(data):
    result = data.split()
    #print(result)
    return result

def file_to_word_ids(data, word_to_id):
    data = [read_words(string) for string in data]
    id_list = []
    for string in data:
        id_list.append([word_to_id[word] for word in string if word in word_to_id])
    return id_list
