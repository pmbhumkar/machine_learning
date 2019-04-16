import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import numpy as np
import pandas as pd
import sys



def pre_process(text):
    text = text.translate(None, string.punctuation)
    processed_text = ""
    text_list = filter(lambda x: x not in stopwords.words('english'), text.split())
    for word in text_list:
        s = SnowballStemmer('english')
        processed_text += s.stem(word) + " "
    return processed_text.strip().lower()


def clean_text(text):
    l = []
    for i in text.split():
        try:
            i.decode('ascii')
            l.append(i)
        except UnicodeDecodeError:
            pass
            
    return ' '.join(l)
        

# text = "Test gap: NetworkHttpPapi tests don't test all my supported versions"

datafile = '../dataset/no_test.csv'
# datafile = 'no_new_bugs'
data = pd.read_csv(datafile, index_col = "Bug ID")
data.head()
data['Summary'] = data['Summary'].apply(clean_text)
data.head()



# datafile = 'no_new_bugs'
test_data = pd.read_csv('../dataset/test.csv', index_col = "Bug ID")
test_data.head()
test_data['Summary'] = test_data['Summary'].apply(clean_text)
test_data.head()

# data = data.rename(columns={"v1":"id", "v2":"text", "v3" : "class"})
# data.head()


data['Summary'] = data['Summary'].astype(str)
data['length'] = data['Summary'].apply(len)
data.head()



test_data['Summary'] = test_data['Summary'].astype(str)
test_data['length'] = test_data['Summary'].apply(len)
test_data.head()


textFeatures = data['Summary'].copy()
textFeatures = textFeatures.apply(pre_process)


test_textFeatures = test_data['Summary'].copy()
test_textFeatures = test_textFeatures.apply(pre_process)

# from sklearn import preprocessing
# label_encode = preprocessing.LabelEncoder()
# st = label_encode.fit_transform(data['Status'])
# data['Status'] = st
from sklearn.ensemble import RandomForestClassifier

features_train, features_test, labels_train, labels_test = train_test_split(textFeatures, data['Status'], test_size=0.05, random_state=10)

from sklearn.pipeline import Pipeline

pipeline = Pipeline([('vect', TfidfVectorizer("english")),
                     ('clf', RandomForestClassifier(n_estimators=20, random_state=0))])
                     
model = pipeline.fit(features_train, labels_train)
print model.score(features_test, labels_test)

print model.predict(test_textFeatures)

# print "Test data"
# print labels_test[labels_test == 'REOPENED']
# print "-" * 80
# # print type(labels_test)
# # print dir(labels_test)


# # print labels_test.where(labels_test == 'REOPENED')
# # print labels_test[labels_test == 'REOPENED']
# reopened_train_bugs_index = []
# for index, dd in enumerate(labels_test):
#     if dd == 'REOPENED':
#         reopened_train_bugs_index.append(index)

# print "Reopened bugs indices"
# print reopened_train_bugs_index
# print "-" * 80

# from sklearn.naive_bayes import MultinomialNB

# mnb = MultinomialNB(alpha=0.2)
# mnb.fit(features_train, labels_train)
# prediction = mnb.predict(features_test)
# prob = mnb.predict_proba(features_test)
# print "Predicted data"
# print [prediction[i] for i in reopened_train_bugs_index]
# print "-" * 80
# reopen_id = np.where(mnb.classes_ == 'REOPENED')


# print "Reopen Percentage"
# for index in reopened_train_bugs_index:
#     print "Reopen percentage : %s" % (prob[index][reopen_id][0] * 100)
# print "-" * 80

# print "Accuracy score : "
# print accuracy_score (
#     labels_test[labels_test == 'REOPENED'],
#     [prediction[i] for i in reopened_train_bugs_index]
# ) * 100




# r = 
# r.fit(features_train, labels_train)
# predict = r.predict(features_test)
# # print dir(predict)
# print labels_test
# print predict

# from sklearn import metrics
# print metrics.mean_absolute_error(labels_test, predict)
# print metrics.mean_squared_error(labels_test, predict)
# print np.sqrt(metrics.mean_squared_error(labels_test, predict))

# print "Accuracy score : "
# print accuracy_score (
#     labels_test,
#     predict
# ) * 100

# Sample test with svm
# from sklearn.svm import LinearSVC

# svc = LinearSVC()
# svc.fit(features_train, labels_train)
# prediction_svc = svc.predict(features_test)


# print accuracy_score (
#     labels_test[labels_test == 'REOPENED'],
#     [prediction_svc[i] for i in reopened_train_bugs_index]
# ) * 100
