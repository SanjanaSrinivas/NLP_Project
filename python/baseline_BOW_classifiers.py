import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report

def get_metrics(y_test, y_predicted):  
    # true positives / (true positives+false positives)
    precision = precision_score(y_test, y_predicted, pos_label=None,
                                    average='weighted')             
    # true positives / (true positives + false negatives)
    recall = recall_score(y_test, y_predicted, pos_label=None,
                              average='weighted')
    
    # harmonic mean of precision and recall
    f1 = f1_score(y_test, y_predicted, pos_label=None, average='weighted')
    
    # true positives + true negatives/ total
    accuracy = accuracy_score(y_test, y_predicted)
    return accuracy, precision, recall, f1


df = pd.read_csv('data/clean_data.csv', header = 0)
data_matrix = df.as_matrix()

_, text_data, labels = data_matrix.T

print "BAG OF WORDS"
count_vect = CountVectorizer(encoding="ISO-8859-1")
count_data = count_vect.fit_transform(text_data)

X_train, X_test, y_train, y_test = train_test_split(count_data, labels, 
                                    test_size=0.33, random_state=324897)

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

clf = SVC()
clf.fit(X_train, y_train)
print "SVM"
y_pred = clf.predict(X_test)
accuracy, precision, recall, f1 = get_metrics(y_test, y_pred)
print("accuracy = %.3f, precision = %.3f, recall = %.3f, f1 = %.3f" % (accuracy, precision, recall, f1))

print "\nTF-IDF"
tfidf_vect = TfidfVectorizer(encoding="ISO-8859-1")
tfidf_data = tfidf_vect.fit_transform(text_data)

X_train, X_test, y_train, y_test = train_test_split(tfidf_data, labels, 
                                    test_size=0.25, random_state=324897)

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

clf = SVC()
clf.fit(X_train, y_train)
print "SVM"
y_pred = clf.predict(X_test)
accuracy, precision, recall, f1 = get_metrics(y_test, y_pred)
print("accuracy = %.3f, precision = %.3f, recall = %.3f, f1 = %.3f" % (accuracy, precision, recall, f1))