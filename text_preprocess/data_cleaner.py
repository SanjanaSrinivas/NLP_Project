import pandas as pd
import numpy as np
import re
import csv
import sys

def clean_text(input_file_path, output_file_path):
    input_file = pd.read_csv(input_file_name, header=1, dtype={"text": str, "label": str})
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
    questions.to_csv(output_file_name, index=False)

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print "Usage: python data_cleaner.py input_file_path output_file_path"
    else:
        input_file_name = sys.argv[1]
        output_file_name = sys.argv[2]
        clean_text(input_file_name, output_file_name)
        print "Clean data file %s created"%(output_file_name)