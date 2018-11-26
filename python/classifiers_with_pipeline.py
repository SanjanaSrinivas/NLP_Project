import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
import nltk
from sklearn import ensemble
from nltk.stem.snowball import SnowballStemmer
import xgboost


# RUN USING python -W ignore flag

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


df = pd.read_csv('./../data/clean_data.csv', header=0)
data_matrix = df.as_matrix()

text_data, labels = data_matrix.T

stemmer = SnowballStemmer("english", ignore_stopwords=True)


class StemmedCountVectorizer(CountVectorizer):
    def build_analyzer(self):
        analyzer = super(StemmedCountVectorizer, self).build_analyzer()
        return lambda doc: ([stemmer.stem(w) for w in analyzer(doc)])


stemmed_count_vect = StemmedCountVectorizer(stop_words='english')

X_train, X_test, y_train, y_test = train_test_split(text_data, labels,
                                                    test_size=0.25, random_state=324897)

# print "***************** LDA ****************"
# lda_model = decomposition.LatentDirichletAllocation(n_components=20, learning_method='online', max_iter=20)


pipeline = Pipeline([('vect', stemmed_count_vect), ('tfidf', TfidfTransformer()),
                     ('clf', MultinomialNB())])
text_clf = pipeline.fit(X_train, y_train)
print "Multinomial NB"
y_pred = text_clf.predict(X_test)
accuracy, precision, recall, f1 = get_metrics(y_test, y_pred)
print("accuracy = %.3f, precision = %.3f, recall = %.3f, f1 = %.3f" % (accuracy, precision, recall, f1))


pipeline1 = Pipeline([('vect', stemmed_count_vect), ('tfidf', TfidfTransformer()),
                      ('clf', LogisticRegression())])
text_clf = pipeline1.fit(X_train, y_train)
print "Logistic Regression"
y_pred = text_clf.predict(X_test)
accuracy, precision, recall, f1 = get_metrics(y_test, y_pred)
print("accuracy = %.3f, precision = %.3f, recall = %.3f, f1 = %.3f" % (accuracy, precision, recall, f1))


pipeline2 = Pipeline([('vect', stemmed_count_vect), ('tfidf', TfidfTransformer()),
                      ('clf', SVC())])
text_clf = pipeline2.fit(X_train, y_train)
print "SVM"
y_pred = text_clf.predict(X_test)
accuracy, precision, recall, f1 = get_metrics(y_test, y_pred)
print("accuracy = %.3f, precision = %.3f, recall = %.3f, f1 = %.3f" % (accuracy, precision, recall, f1))


pipeline3 = Pipeline([('vect', stemmed_count_vect), ('tfidf', TfidfTransformer()),
                      ('clf', ensemble.RandomForestClassifier())])
text_clf = pipeline3.fit(X_train, y_train)
print "Random forest"
y_pred = text_clf.predict(X_test)
accuracy, precision, recall, f1 = get_metrics(y_test, y_pred)
print("accuracy = %.3f, precision = %.3f, recall = %.3f, f1 = %.3f" % (accuracy, precision, recall, f1))


pipeline4 = Pipeline([('vect', stemmed_count_vect), ('tfidf', TfidfTransformer()),
                      ('clf', xgboost.XGBClassifier())])
text_clf = pipeline4.fit(X_train, y_train)
print "XGBOOST"
y_pred = text_clf.predict(X_test)
accuracy, precision, recall, f1 = get_metrics(y_test, y_pred)
print("accuracy = %.3f, precision = %.3f, recall = %.3f, f1 = %.3f" % (accuracy, precision, recall, f1))

