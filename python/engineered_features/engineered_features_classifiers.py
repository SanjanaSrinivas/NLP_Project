import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
import tenses
import named_entities
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

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


df = pd.read_csv('../data/total_labelled.csv', header=0, encoding="utf-8")
df.columns=['text', 'label']
df["text"] = df["text"].str.replace(r"[^A-Za-z0-9(),!?@\_\n]", " ")
print df.head()
data_matrix = df.as_matrix()

text_data, labels = data_matrix.T

count_vect = CountVectorizer(encoding="ISO-8859-1")
count_data = count_vect.fit_transform(text_data)

# tfidf_vect = TfidfVectorizer(encoding="ISO-8859-1")
# tfidf_data = tfidf_vect.fit_transform(text_data)
# print type(tfidf_data)
# print tfidf_data.shape


X_tense = np.array(tenses.get_tense_vectors(text_data))

X_ner = np.array(named_entities.get_ner_vectors(text_data))

X = np.concatenate((count_data.toarray(), X_tense, X_ner), axis=1)
print X[:5]

X_train, X_test, y_train, y_test = train_test_split(X, labels,
                                                    test_size=0.33, random_state=324897)

clf = MultinomialNB()
clf.fit(X_train, y_train)
print "Multinomial NB"
y_pred = clf.predict(X_test)
accuracy, precision, recall, f1 = get_metrics(y_test, y_pred)
print("accuracy = %.3f, precision = %.3f, recall = %.3f, f1 = %.3f" % (accuracy, precision, recall, f1))
print(classification_report(y_test, y_pred))

clf = LogisticRegression()
clf.fit(X_train, y_train)
print "Logistic Regression"
y_pred = clf.predict(X_test)
accuracy, precision, recall, f1 = get_metrics(y_test, y_pred)
print("accuracy = %.3f, precision = %.3f, recall = %.3f, f1 = %.3f" % (accuracy, precision, recall, f1))
print(classification_report(y_test, y_pred))

clf = SVC(kernel='linear')
clf.fit(X_train, y_train)
print "SVM"
y_pred = clf.predict(X_test)
accuracy, precision, recall, f1 = get_metrics(y_test, y_pred)
print("accuracy = %.3f, precision = %.3f, recall = %.3f, f1 = %.3f" % (accuracy, precision, recall, f1))
print(classification_report(y_test, y_pred))
