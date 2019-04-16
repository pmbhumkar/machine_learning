import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
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
    return processed_text.strip()


# text = "Test gap: NetworkHttpPapi tests don't test all my supported versions"

datafile = '../dataset/no_new.csv'
data = pd.read_csv(datafile, index_col = "Bug ID")
data.head()

# data.drop(['NEW'], inplace=True)

feature_encode = []
label_encode = preprocessing.LabelEncoder()
for feature in ['Component','Reporter','Assignee']:
    feature_encode.append(label_encode.fit_transform(data[feature]))

# print feature_encode
combined_feature = zip(feature_encode[0], feature_encode[1], feature_encode[2])
# print combined_feature

st = label_encode.fit_transform(data['Status'])
data['Status'] = st
# print data['Status']

reopen_id = np.where(label_encode.classes_ == 'REOPENED')
# print reopen_id
# print reopen_id[0][0]
# print '-'*80
# print label_encode.classes_[reopen_id]
# print '-'*80


# sys.exit(0)

features_train, features_test, labels_train, labels_test = train_test_split(combined_feature, data['Status'], test_size=0.05, random_state=10)
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

from sklearn.naive_bayes import MultinomialNB

# mnb = MultinomialNB(alpha=0.2)
# mnb.fit(features_train, labels_train)
# prediction = mnb.predict(features_test)
# prob = mnb.predict_proba(features_test)
# print "Predicted data"
# print [prediction[i] for i in reopened_train_bugs_index]
# print "-" * 80
# reopen_id = np.where(mnb.classes_ == 'REOPENED')





from sklearn.ensemble import RandomForestClassifier

# print help(RandomForestClassifier)
r = RandomForestClassifier(n_estimators=4, random_state=0)
r.fit(features_train, labels_train)
# print help(r.predict)
predict = r.predict(features_test)
prob = r.predict_proba(features_test)
# print dir(predict)
reopened_train_bugs_index = []
for index, dd in enumerate(labels_test):
    if dd == reopen_id[0][0]:
        reopened_train_bugs_index.append(index)


print "Reopen Probability"
for index in reopened_train_bugs_index:
    print "Bug [%s]: %s" % (labels_train.index.values[index], prob[index][reopen_id][0])

print "-" * 80


# print labels_test
# print predict

# from sklearn import metrics
# print metrics.mean_absolute_error(labels_test, predict)
# print metrics.mean_squared_error(labels_test, predict)
# print np.sqrt(metrics.mean_squared_error(labels_test, predict))

print "Accuracy score : %s" % (
    accuracy_score (
        labels_test[labels_test == reopen_id[0][0]],
        [predict[i] for i in reopened_train_bugs_index]
        # labels_test,
        # predict
    ) * 100
)

print "Classes: "
for index, class_name in enumerate(label_encode.classes_):
    print "%s : %s" % (index, class_name)


import matplotlib

import matplotlib.pyplot as plt
# matplotlib.use('Agg')
import scikitplot as skplt
# print help(skplt.metrics.plot_confusion_matrix)
skplt.metrics.plot_confusion_matrix(labels_test, predict, normalize = True)
plt.show()

# # Sample test with svm
# from sklearn.svm import LinearSVC

# svc = LinearSVC()
# svc.fit(features_train, labels_train)
# prediction_svc = svc.predict(features_test)
# print prediction_svc

# print accuracy_score (
#     labels_test[labels_test == 'REOPENED'],
#     [prediction_svc[i] for i in reopened_train_bugs_index]
# ) * 100

