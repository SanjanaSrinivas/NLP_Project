import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
df = pd.read_csv('data/clean_data.csv', header = 0)
data_matrix = df.as_matrix()

_, text_data, labels = data_matrix.T
count_vect = CountVectorizer(encoding = "ISO-8859-1")
count_data = count_vect.fit_transform(text_data)

X_train, X_test, y_train, y_test = train_test_split(count_data, labels, 
                                    test_size = 0.33, random_state=324897)

clf = MultinomialNB()
clf.fit(X_train, y_train)
print "Multinomial NB"
print clf.score(X_test, y_test)

clf = LogisticRegression()
clf.fit(X_train, y_train)
print "Logistic Regression"
print clf.score(X_test, y_test)

clf = SVC()
clf.fit(X_train, y_train)
print "SVM"
print clf.score(X_test, y_test)