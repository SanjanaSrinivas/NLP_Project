import nltk
import pandas as pd
import numpy as np
import re
import codecs
import csv
from nltk.tokenize import RegexpTokenizer
import matplotlib.pyplot as plt
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

input_file = pd.read_csv("data/total_labelled.csv", header=1, dtype={"text": str, "label": str})
input_file.columns=['text', 'label']
input_file['label'] = input_file['label'].fillna(' NA')
def standardize_text(df, text_field):
    df[text_field] = df[text_field].str.replace(r"#", "")
    df[text_field] = df[text_field].str.replace(r"http\S+", "")
    df[text_field] = df[text_field].str.replace(r"http", "")
    df[text_field] = df[text_field].str.replace(r"@\S+", "")
    df[text_field] = df[text_field].str.replace(r"[^A-Za-z0-9(),?@\_\n ]", " ")
    df[text_field] = df[text_field].str.replace(r"@", "at")
    df[text_field] = df[text_field].str.replace(r"\s+", " ")
    df[text_field] = df[text_field].str.lower()
    return df

pd.set_option('display.max_colwidth', -1)

questions = standardize_text(input_file, "text")
questions.to_csv("data/clean_data.csv", index=False)
clean_questions = pd.read_csv("data/clean_data.csv")

#print clean_questions.groupby("class_label").count()

clean_questions = pd.read_csv("data/clean_data.csv")
tokenizer = RegexpTokenizer(r'\w+')

clean_questions["tokens"] = clean_questions["text"].apply(tokenizer.tokenize)
print clean_questions.head()

all_words = [word for tokens in clean_questions["tokens"] for word in tokens]
sentence_lengths = [len(tokens) for tokens in clean_questions["tokens"]]
VOCAB = sorted(list(set(all_words)))
print("%s words total, with a vocabulary size of %s" % (len(all_words), len(VOCAB)))
print("Max sentence length is %s" % max(sentence_lengths))

