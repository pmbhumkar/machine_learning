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
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer

# class CustomEncode(preprocessing.LabelEncoder):
#     def fit(self, X, y = None, **fit_params):
#         return preprocessing.LabelEncoder().fit(X).reshape(-1, 1)
#     def fit_transform(self, X, y = None):
#         return preprocessing.LabelEncoder().fit(X).transform(X).reshape(-1, 1)
#     # def fit_transform(self, X, y = None):
#     #     return super(CustomEncode).fit_transform(X)


class CustomEncode(preprocessing.LabelEncoder):
    def fit(self, X, y = None, **fit_params):
        return super(CustomEncode, self).fit(X).reshape(-1, 1)
    def fit_transform(self, X, y = None):
        return super(CustomEncode, self).fit_transform(X).reshape(-1, 1)
    def transform(self, X, y = None):
        return super(CustomEncode, self).transform(X).reshape(-1, 1)
    

class MultiColumnLabelEncoder:
    
    def __init__(self, columns = None):
        self.columns = columns # list of column to encode
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        '''
        Transforms columns of X specified in self.columns using
        LabelEncoder(). If no columns specified, transforms all
        columns in X.
        '''
        
        output = X.copy()
        
        if self.columns is not None:
            for col in self.columns:
                output[col] = preprocessing.LabelEncoder().fit_transform(output[col])
        else:
            for colname, col in output.iteritems():
                output[colname] = preprocessing.LabelEncoder().fit_transform(col)
        
        return output
    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)



    
def pre_process(text):
    text = text.translate(None, string.punctuation)
    processed_text = ""
    text_list = filter(lambda x: x not in stopwords.words('english'), text.split())
    for word in text_list:
        s = SnowballStemmer('english')
        processed_text += s.stem(word) + " "
    return processed_text.strip()


# text = "Test gap: NetworkHttpPapi tests don't test all my supported versions"

datafile = '../dataset/meta_train.csv'
data = pd.read_csv(datafile, index_col = "Bug ID")
data.head()


datafile = '../dataset/user_test.csv'
test_data = pd.read_csv(datafile, index_col = "Bug ID")
test_data.head()

# print data[['Summary', 'Assignee']]

# data.drop(['NEW'], inplace=True)
# l = preprocessing.OneHotEncoder()
# print l.fit_transform(data)
# print feature_encode
# combined_feature = zip(data['Component'], data['Reporter'], data['Assignee'])
# print combined_feature
# print combined_feature

features_train, features_test, labels_train, labels_test = train_test_split(data[['Reporter', 'Component', 'Assignee']], data['Status'], test_size=0.01, random_state=10)


# st = label_encode.fit_transform(data['Status'])
# data['Status'] = st
# print data['Status']

# reopen_id = np.where(label_encode.classes_ == 'REOPENED')
# print reopen_id
# print reopen_id[0][0]
# print '-'*80
# print label_encode.classes_[reopen_id]
# print '-'*80


# sys.exit(0)
from sklearn.ensemble import RandomForestClassifier

from sklearn.pipeline import Pipeline, FeatureUnion
pipeline = Pipeline([
    # ('cv', CountVectorizer(max_features = 3)),
    ('ml', MultiColumnLabelEncoder()),
    ('rfc', RandomForestClassifier(n_estimators=6, random_state=0))])
                     
model = pipeline.fit(features_train, labels_train)

prob = model.predict_proba(test_data[['Reporter', 'Component', 'Assignee']])
print prob
# reopened_train_bugs_index = []
# for index, dd in enumerate(test_data['Status']):
#     if dd == 'REOPENED':
#         reopened_train_bugs_index.append(index)


print "Reopen Probability"

reopen_id = np.where(model.classes_ == 'REOPENED')
for ind, bug in enumerate(prob):
    print "Bug [%s]: %s" % (test_data.index.values[ind], bug[reopen_id][0])

print "-" * 80

print model.score(features_test, labels_test)



sys.exit(0)
# feature_encode = []
# label_encode = preprocessing.LabelEncoder()
# for feature in ['Component','Reporter','Assignee']:
#     feature_encode.append(label_encode.fit_transform(data[feature]))





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







# print help(RandomForestClassifier)
r = RandomForestClassifier(n_estimators=15, random_state=0)
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
print predict

# from sklearn import metrics
# print metrics.mean_absolute_error(labels_test, predict)
# print metrics.mean_squared_error(labels_test, predict)
# print np.sqrt(metrics.mean_squared_error(labels_test, predict))

print "Accuracy score : "
print accuracy_score (
    labels_test[labels_test == reopen_id[0][0]],
    [predict[i] for i in reopened_train_bugs_index]
    # labels_test,
    # predict
) * 100



import matplotlib.pyplot as plt

import scikitplot as skplt
print help(skplt.metrics.plot_confusion_matrix)
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

