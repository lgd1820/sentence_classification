'''
작성일 : 2020-09-11
작성자 : 이권동
코드 개요 : 문장들에 대해서 불필요한 문자를 제거하고 형태소 단위로 분리시키는 코드
형태소로 분리된 단어는 정수의 키와 매핑되어 voca 라는 단어 사전에 저장
'''
from konlpy.tag import Kkma
from keras.utils import *
import numpy as np
import json
import re

'''
    함수 개요 :
        문자열을 입력하면 불필요한 문자들을 제거 후 형태소 단위로 분리시키는 함수
    매개변수 :
        string = 문자열
'''
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

'''
    함수 개요 :
        intent가 저장된 위치의 리스트들을 입력하면 문장별로 끊는 함수
        clean_str 을 통해 형태소 단위로 분리된 단어를 반환
    매개변수 :
        intent_list=인텐트가 저장된 경로 리스트
'''
def load_data_and_labels(intent_list):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
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
    return [x_text, y]

'''
    함수 개요 :
        형태소 단위로 분리된 문장을 입력받아 단어 사전에 등록하는 함수
        만약 사전에 등록되어있으면 단어를 숫자로 변환
    매개변수 :
        sentence = 문장
        max_length = 현재 모든 문장 중 제일 긴 문장의 단어 수
        voca = 단어 사전
'''
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
