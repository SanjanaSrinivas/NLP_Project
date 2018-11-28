import numpy as np
import pandas as pd
import sys
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
from nltk.tokenize import RegexpTokenizer
import gensim

# RUN USING python -W ignore flag

def get_metrics(y_test, y_predicted):  
    # true positives / (true positives+false positives)
    precision = precision_score(y_test, y_predicted, pos_label=' A',
                                    average='weighted')             
    # true positives / (true positives + false negatives)
    recall = recall_score(y_test, y_predicted, pos_label=' A',
                              average='weighted')
    
    # harmonic mean of precision and recall
    f1 = f1_score(y_test, y_predicted, pos_label=' A', average='weighted')
    
    # true positives + true negatives/ total
    accuracy = accuracy_score(y_test, y_predicted)
    return accuracy, precision, recall, f1

def run_classifiers(input_file_path):
    df = pd.read_csv(input_file_path, header = 1)
    df.columns=['text', 'label']
    df['label'] = df['label'].fillna(' NA')
    #data_matrix = df.to_records(index=False)
    #data_matrix = np.asarray(data_matrix)
    data_matrix = df.values
    text_data, labels = data_matrix.T

    print "BAG OF WORDS"
    count_vect = CountVectorizer()
    count_data = count_vect.fit_transform(text_data)

    X_train, X_test, y_train, y_test = train_test_split(count_data, labels, 
                                        test_size=0.2, random_state=40)

    clf = DummyClassifier()
    clf.fit(X_train, y_train)
    print "Baseline"
    y_pred = clf.predict(X_test)
    accuracy, precision, recall, f1 = get_metrics(y_test, y_pred)
    print("accuracy = %.3f, precision = %.3f, recall = %.3f, f1 = %.3f" % (accuracy, precision, recall, f1))


    print "\nTF-IDF"
    tfidf_vect = TfidfVectorizer()
    tfidf_data = tfidf_vect.fit_transform(text_data)

    X_train, X_test, y_train, y_test = train_test_split(tfidf_data, labels, 
                                        test_size=0.2, random_state=42)

    clf = RandomForestClassifier(n_estimators=100, class_weight={' NA': 4})
    clf.fit(X_train, y_train)
    print "Random Forest"
    y_pred = clf.predict(X_test)
    accuracy, precision, recall, f1 = get_metrics(y_test, y_pred)
    print("accuracy = %.3f, precision = %.3f, recall = %.3f, f1 = %.3f" % (accuracy, precision, recall, f1))

    clf = MultinomialNB()
    clf.fit(X_train, y_train)
    print "Multinomial NB"
    y_pred = clf.predict(X_test)
    accuracy, precision, recall, f1 = get_metrics(y_test, y_pred)
    print("accuracy = %.3f, precision = %.3f, recall = %.3f, f1 = %.3f" % (accuracy, precision, recall, f1))

    clf = LogisticRegression()
    clf.fit(X_train, y_train)
    print "Logistic Regression"
    y_pred = clf.predict(X_test)
    accuracy, precision, recall, f1 = get_metrics(y_test, y_pred)
    print("accuracy = %.3f, precision = %.3f, recall = %.3f, f1 = %.3f" % (accuracy, precision, recall, f1))

    clf = SVC(kernel = 'sigmoid', C=70, gamma=0.05)
    clf.fit(X_train, y_train)
    print "SVM"
    y_pred = clf.predict(X_test)
    accuracy, precision, recall, f1 = get_metrics(y_test, y_pred)
    print("accuracy = %.3f, precision = %.3f, recall = %.3f, f1 = %.3f" % (accuracy, precision, recall, f1))

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print "Usage: python -W ignore final_classifiers.py input_file_path"
        # Ignoring future warnings to prettify output
    else:
        input_file_name = sys.argv[1]
        run_classifiers(input_file_name)