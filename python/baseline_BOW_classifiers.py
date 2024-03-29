import numpy as np
import pandas as pd
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


df = pd.read_csv('data/clean_data.csv', header = 1)
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

clf = MultinomialNB()
clf.fit(X_train, y_train)
print "Multinomial NB"
y_pred = clf.predict(X_test)
accuracy, precision, recall, f1 = get_metrics(y_test, y_pred)
print("accuracy = %.3f, precision = %.3f, recall = %.3f, f1 = %.3f" % (accuracy, precision, recall, f1))

clf = RandomForestClassifier()
clf.fit(X_train, y_train)
print "Random Forest"
y_pred = clf.predict(X_test)
accuracy, precision, recall, f1 = get_metrics(y_test, y_pred)
print("accuracy = %.3f, precision = %.3f, recall = %.3f, f1 = %.3f" % (accuracy, precision, recall, f1))

clf = LogisticRegression()
clf.fit(X_train, y_train)
print "Logistic Regression"
y_pred = clf.predict(X_test)
accuracy, precision, recall, f1 = get_metrics(y_test, y_pred)
print("accuracy = %.3f, precision = %.3f, recall = %.3f, f1 = %.3f" % (accuracy, precision, recall, f1))

clf = SVC(kernel='linear', gamma='scale')
clf.fit(X_train, y_train)
print "SVM"
y_pred = clf.predict(X_test)
accuracy, precision, recall, f1 = get_metrics(y_test, y_pred)
print("accuracy = %.3f, precision = %.3f, recall = %.3f, f1 = %.3f" % (accuracy, precision, recall, f1))

clf = DummyClassifier()
clf.fit(X_train, y_train)
print "Dummy"
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

clf = DummyClassifier()
clf.fit(X_train, y_train)
print "Dummy"
y_pred = clf.predict(X_test)
accuracy, precision, recall, f1 = get_metrics(y_test, y_pred)
print("accuracy = %.3f, precision = %.3f, recall = %.3f, f1 = %.3f" % (accuracy, precision, recall, f1))

'''
print "\nWord2Vec"
word2vec_path = "GoogleNews-vectors-negative300.bin.gz"
word2vec = gensim.models.KeyedVectors.load_word2vec_format(word2vec_path, binary=True)

def get_average_word2vec(tokens_list, vector, generate_missing=False, k=300):
    if len(tokens_list)<1:
        return np.zeros(k)
    if generate_missing:
        vectorized = [vector[word] if word in vector else np.random.rand(k) for word in tokens_list]
    else:
        vectorized = [vector[word] if word in vector else np.zeros(k) for word in tokens_list]
    length = len(vectorized)
    summed = np.sum(vectorized, axis=0)
    averaged = np.divide(summed, length)
    return averaged

def get_word2vec_embeddings(vectors, clean_questions, generate_missing=False):
    embeddings = clean_questions['tokens'].apply(lambda x: get_average_word2vec(x, vectors, generate_missing=generate_missing))
    return list(embeddings)

clean_questions = pd.read_csv("data/clean_data.csv")
tokenizer = RegexpTokenizer(r'\w+')
clean_questions["tokens"] = clean_questions["text"].apply(tokenizer.tokenize)

embeddings = get_word2vec_embeddings(word2vec, clean_questions)
X_train, X_test, y_train, y_test = train_test_split(embeddings, labels, test_size=0.2, random_state=40)

clf = RandomForestClassifier()
clf.fit(X_train, y_train)
print "Random Forest"
y_pred = clf.predict(X_test)
accuracy, precision, recall, f1 = get_metrics(y_test, y_pred)
print("accuracy = %.3f, precision = %.3f, recall = %.3f, f1 = %.3f" % (accuracy, precision, recall, f1))

clf = LogisticRegression()
clf.fit(X_train, y_train)
print "Logistic Regression"
y_pred = clf.predict(X_test)
accuracy, precision, recall, f1 = get_metrics(y_test, y_pred)
print("accuracy = %.3f, precision = %.3f, recall = %.3f, f1 = %.3f" % (accuracy, precision, recall, f1))

clf = SVC(kernel='linear', gamma='scale')
clf.fit(X_train, y_train)
print "SVM"
y_pred = clf.predict(X_test)
accuracy, precision, recall, f1 = get_metrics(y_test, y_pred)
print("accuracy = %.3f, precision = %.3f, recall = %.3f, f1 = %.3f" % (accuracy, precision, recall, f1))
'''