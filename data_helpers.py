from konlpy.tag import Kkma
from keras.utils import *
import numpy as np
import json
import re


def clean_str(string):
    string = re.sub(r"[-=+,#/\?:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\'…》]", "", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    kkma = Kkma()
    #print(string, len(string))
    if len(string) == 1:
        return ""
    string = " ".join([word[0] for word in kkma.pos(string) if word[1][0] != "J"])
    #print(string.strip().lower())
    #string = '<start> ' + string + ' <end>'
    return string.strip().lower()


def load_data_and_labels(intent_list):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    #positive_examples = list(open(positive_data_file, "r", encoding='utf-8').readlines())
    #positive_examples = [s.strip() for s in positive_examples]
    #negative_examples = list(open(negative_data_file, "r", encoding='utf-8').readlines())
    #negative_examples = [s.strip() for s in negative_examples]
    examples_list = []
    for il in intent_list:
        example = list(open(il, "r", encoding='utf-8').readlines())
        example = [s.replace("\xa0", "").strip() for s in example]
        examples_list.append(example)
    # Split by words
    x_text = sum(examples_list, [])
    x_text = [clean_str(sent) for sent in x_text]
    # Generate labels
    labels = []
    for one_hot in range(len(examples_list)):
        label = [one_hot for _ in examples_list[one_hot]]
        labels.append(label)
    labels = sum(labels, [])
    y = np_utils.to_categorical(labels, len(examples_list))
    #positive_labels = [[0, 1] for _ in positive_examples]
    #negative_labels = [[1, 0] for _ in negative_examples]
    #y = np.concatenate([positive_labels, negative_labels], 0)
    return [x_text, y]


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]

def word_map(sentence, max_length, voca):
    word_map_int = []
    for idx in range(max_length):
        if idx + 1 > len(sentence):
            word_map_int.append(0)
        else:
            if sentence[idx] in voca:
                word_map_int.append(voca[sentence[idx]])
            else:
                voca[sentence[idx]] = len(voca)
                word_map_int.append(voca[sentence[idx]])
                with open("data/voca/voca", "w", encoding="utf-8") as f:
                    json.dump(voca, f, ensure_ascii=False)
                    f.close()
    return word_map_int
