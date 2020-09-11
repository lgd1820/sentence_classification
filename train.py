'''
작성일 : 2020-09-11
작성자 : 이권동
코드 개요 : 인텐트를 CNN으로 학습시켜 문장분류를 하는 코드
'''

import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers

from keras import layers
from keras.models import Sequential, Model
from keras.preprocessing.text import Tokenizer, one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout, Flatten, MaxPooling1D, Input, Concatenate
from keras.models import load_model
from keras.utils import *
from keras import backend as K

#from text_cnn import TextCNN
from absl import flags, app
import gensim
from gensim.models import Word2Vec
import ast
import json

import time

'''
    함수 개요 :
        데이터 전처리를 위한 함수
'''
def preprocess():
    # Data Preparation
    # ==================================================

    # Load data
    print("Loading data...")
    #intent_list = ["./data/forest/only_book/book", "./data/forest/only_book/etc"]
    #intent_list = ["./data/forest/only_eat/eat", "./data/forest/only_eat/etc"]
    #intent_list = ["./data/forest/only_enjoy/enjoy", "./data/forest/only_enjoy/etc"]
    intent_list = ["./data/forest/only_facility/facility", "./data/forest/only_facility/etc"]
    x_text, y = data_helpers.load_data_and_labels(intent_list)

    # 문장 최대 길이
    max_document = open("data/eval/ym", "r")
    sentences = max_document.readlines()
    texts = [data_helpers.clean_str(sentence) for sentence in sentences]
    max_document_length = max([len(text.split(" ")) for text in texts])

    del sentences, texts

    # load 사전
    voca_read = open("data/voca/voca", "r")
    vocab = json.load(voca_read)
    voca_read.close()

    x = [data_helpers.word_map(x.split(" "),max_document_length, vocab) for x in x_text]
    x = np.array(x)

    # Randomly shuffle data
    np.random.seed(10) # 난수 생성 패턴
    shuffle_indices = np.random.permutation(np.arange(len(y))) # shuffle
    x_shuffled = x[shuffle_indices]
    y_shuffled = y[shuffle_indices]

    # Split train/test set
    # TODO: This is very crude, should use cross-validation
    dev_sample_index = -1 * int(0.8 * len(y))
    x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
    y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]

    #x_train, x_dev = x_shuffled, x_shuffled[dev_sample_index:]
    #y_train, y_dev = y_shuffled, y_shuffled[dev_sample_index:]

    m = Word2Vec([x.split(" ") for x in x_text], size=128)
    embeddings_index = {}
    for i in range(len(m.wv.vocab)):
        word = list(m.wv.vocab)[i]
        coefs = m[word]
        embeddings_index[word] = coefs

    
    embedding_matrix = np.zeros((len(vocab), 128))
    
    for word, i in vocab.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    np.save("data/voca/word2vec", embedding_matrix)

    del x, y, x_shuffled, y_shuffled

    print("Vocabulary Size: {:d}".format(len(vocab)))
    print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))

    return x_train, y_train, vocab, x_dev, y_dev, embedding_matrix

'''
    함수 개요 :
        recall 을 계산하는 함수
    매개변수 :
        y_target = 레이블 값
        y_pred = 예측한 레이블 값
'''
def recall(y_target, y_pred):
    # clip(t, clip_value_min, clip_value_max) : clip_value_min~clip_value_max 이외 가장자리를 깎아 낸다
    # round : 반올림한다
    y_target_yn = K.round(K.clip(y_target, 0, 1)) # 실제값을 0(Negative) 또는 1(Positive)로 설정한다
    y_pred_yn = K.round(K.clip(y_pred, 0, 1)) # 예측값을 0(Negative) 또는 1(Positive)로 설정한다

    # True Positive는 실제 값과 예측 값이 모두 1(Positive)인 경우이다
    count_true_positive = K.sum(y_target_yn * y_pred_yn) 

    # (True Positive + False Negative) = 실제 값이 1(Positive) 전체
    count_true_positive_false_negative = K.sum(y_target_yn)

    # Recall =  (True Positive) / (True Positive + False Negative)
    # K.epsilon()는 'divide by zero error' 예방차원에서 작은 수를 더한다
    recall = count_true_positive / (count_true_positive_false_negative + K.epsilon())

    # return a single tensor value
    return recall


'''
    함수 개요 :
        precision 을 계산하는 함수
    매개변수 :
        y_target = 레이블 값
        y_pred = 예측한 레이블 값
'''
def precision(y_target, y_pred):
    # clip(t, clip_value_min, clip_value_max) : clip_value_min~clip_value_max 이외 가장자리를 깎아 낸다
    # round : 반올림한다
    y_pred_yn = K.round(K.clip(y_pred, 0, 1)) # 예측값을 0(Negative) 또는 1(Positive)로 설정한다
    y_target_yn = K.round(K.clip(y_target, 0, 1)) # 실제값을 0(Negative) 또는 1(Positive)로 설정한다

    # True Positive는 실제 값과 예측 값이 모두 1(Positive)인 경우이다
    count_true_positive = K.sum(y_target_yn * y_pred_yn) 

    # (True Positive + False Positive) = 예측 값이 1(Positive) 전체
    count_true_positive_false_positive = K.sum(y_pred_yn)

    # Precision = (True Positive) / (True Positive + False Positive)
    # K.epsilon()는 'divide by zero error' 예방차원에서 작은 수를 더한다
    precision = count_true_positive / (count_true_positive_false_positive + K.epsilon())

    # return a single tensor value
    return precision

'''
    함수 개요 :
        f1 score 를 계산하는 함수
    매개변수 :
        y_target = 레이블 값
        y_pred = 예측한 레이블 값
'''
def f1score(y_target, y_pred):
    _recall = recall(y_target, y_pred)
    _precision = precision(y_target, y_pred)
    # K.epsilon()는 'divide by zero error' 예방차원에서 작은 수를 더한다
    _f1score = ( 2 * _recall * _precision) / (_recall + _precision+ K.epsilon())
    
    # return a single tensor value
    return _f1score

'''
    함수 개요 :
        학습 네트워크 설정
    매개변수 :
       max_len = 문장 최대 길이
       num_class = 레이블 수
       vocab = 단어 사전
       embedding_dim = 워드 임베딩 된 벡터 차원 수
       filter_sizes = 필터 크기
       embedding_matrix = 워드 임베딩 된 벡터
'''
def set_model(max_len, num_class, vocab, embedding_dim, filter_sizes, embedding_matrix):
    sequence_input = Input(shape=(max_len,), dtype='int32')
    
    embedding_layer = Embedding(len(vocab),
                                embedding_dim,
                                weights=[embedding_matrix],
                                input_length=max_len,
                                trainable=False)

    embedded_sequences = embedding_layer(sequence_input)

    convs = []

    for fsz in filter_sizes:
        x = Conv1D(128, fsz, activation='relu',padding='same')(embedded_sequences)
        #x = MaxPooling1D(pool_size=(max_len - fsz + 1), strides=1)(x)
        x = MaxPooling1D()(x)
        convs.append(x)
        
    x = Concatenate(axis=-1)(convs)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    #output = Dense(num_class, activation='softmax')(x)
    output = Dense(2, activation='sigmoid')(x)

    model = Model(sequence_input, output)
    #model.compile(loss='categorical_crossentropy',
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  #metrics=['accuracy'])
                  metrics=['accuracy', precision, recall, f1score]) 
                  #metrics=['accuracy', ])
    model.summary() 
    return model

#def train(x_train, y_train, vocab, x_dev, y_dev):
    # Training
    # ==================================================
    


def main(argv=None):
    x_train, y_train, vocab, x_dev, y_dev, embedding_matrix = preprocess()
    co = 0

    for y_d in y_dev:
        if y_d[0] == 1.0:
            co += 1

    print(co, len(y_dev), len(y_train))

    model = set_model(x_train.shape[1], y_train.shape[1], vocab, 128, FLAGS.filter_sizes, embedding_matrix)
    history = model.fit(
        x_train, y_train,
        epochs=50,
        verbose=True,
        validation_data=(x_dev, y_dev),
        batch_size=10)
    #loss, accuracy = model.evaluate(x_train, y_train, verbose=True)
    #print("Training {:.4f}".format(accuracy))
    #loss, accuracy = model.evaluate(x_dev, y_dev, verbose=False)
    #print("Testing {:.4f}".format(accuracy))
    
    
    print(model.evaluate(x_train, y_train, verbose=False))
    print(model.evaluate(x_dev, y_dev, verbose=False))

    #np.save("a", y_dev)
    #model_json = model.to_json()

    
    #now = str(int(time.time()))
    #with open("data/model/" + now + "_model", "w") as md:
    #    md.write(model_json)
    #model.save_weights("data/model/" + now + "_weight")
    #print(now)
    
if __name__ == '__main__':
    print("main")    
    app.run(main)
