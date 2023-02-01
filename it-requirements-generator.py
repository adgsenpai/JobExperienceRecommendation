
# # Job Recommendations
#
# This notebook creates a model, to recommend job positions given a position requirements description . This is done only for IT jobs.

from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

import nltk
from nltk.corpus import stopwords
from sklearn.preprocessing import LabelEncoder
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, auc, roc_curve, roc_auc_score

data = pd.read_csv('data.csv')
print(data.columns)
# selecting only IT Jobs
df = data
# selecting
cols = ['RequiredQual', 'Title', 'JobDescription', 'JobRequirement']
df = df[cols]
df.head(5)


# # Modifying Job Titles
# Selecting only top 21 job titles, to manage class imbalance

classes = df['Title'].value_counts()[:21]
keys = classes.keys().to_list()

df = df[df['Title'].isin(keys)]
df['Title'].value_counts()


# Change job titles to base title. For example, chaning Senior Java Developer to Java Developer.

def chane_titles(x):
    x = x.strip()
    if x == 'Senior Java Developer':
        return 'Java Developer'
    elif x == 'Senior Software Engineer':
        return 'Software Engineer'
    elif x == 'Senior QA Engineer':
        return 'Software QA Engineer'
    elif x == 'Senior Software Developer':
        return 'Senior Web Developer'
    elif x == 'Senior PHP Developer':
        return 'PHP Developer'
    elif x == 'Senior .NET Developer':
        return '.NET Developer'
    elif x == 'Senior Web Developer':
        return 'Web Developer'
    elif x == 'Database Administrator':
        return 'Database Admin/Dev'
    elif x == 'Database Developer':
        return 'Database Admin/Dev'

    else:
        return x


df['Title'] = df['Title'].apply(chane_titles)
df['Title'].value_counts()


# # Building custom tokenizer to process text


class LemmaTokenizer(object):
    def __init__(self):
        # lemmatize text - convert to base form
        self.wnl = WordNetLemmatizer()
        # creating stopwords list, to ignore lemmatizing stopwords
        self.stopwords = stopwords.words('english')

    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc) if t not in self.stopwords]


# removing new line characters, and certain hypen patterns
# convert all df['RequiredQual'] to string
df['RequiredQual'] = df['RequiredQual'].astype(str)
df['RequiredQual'] = df['RequiredQual'].apply(lambda x: x.replace(
    '\n', ' ').replace('\r', '').replace('- ', ''). replace(' - ', ' to '))


# # Featurizing Text

# train features and labels
y = df['Title']
X = df['RequiredQual']
# tdif feature rep
vectorizer = TfidfVectorizer(tokenizer=LemmaTokenizer(), stop_words='english')
vectorizer.fit(X)
# transoforming text to tdif features
tfidf_matrix = vectorizer.transform(X)
# sparse matrix to dense matrix for training
X_tdif = tfidf_matrix.toarray()
# encoding text labels in categories
enc = LabelEncoder()
enc.fit(y.values)
y_enc = enc.transform(y.values)

X_train_words, X_test_words, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.15, random_state=10)

X_train = vectorizer.transform(X_train_words)
X_train = X_train.toarray()

X_test = vectorizer.transform(X_test_words)
X_test = X_test.toarray()


# # Training Naive Bayes
# Looks pretty overfit

gnb = GaussianNB()
train_preds = gnb.fit(X_train, y_train).predict(X_train)
test_preds = gnb.predict(X_test)

print('Train acc: {0}'.format(accuracy_score(y_train, train_preds)))
print('Test acc: {0}'.format(accuracy_score(y_test, test_preds)))


# # Training Logistic Regression
# By modifiying the maximum number of iterations, and regularization, C, the above experienced overfitting was reduced significantly
#


logistic = LogisticRegression(max_iter=15, verbose=1, C=0.75)

train_preds = logistic.fit(X_train, y_train).predict(X_train)
test_preds = logistic.predict(X_test)

print('Train acc: {0}'.format(accuracy_score(y_train, train_preds)))
print('Test acc: {0}'.format(accuracy_score(y_test, test_preds)))


# # Creating Job Recommendations
# Recommends 2 job position alternatives given a job requirement. By obtaining probability of class predictions, and picking the top N predictions, other than true label, N closest recommendations can be got

preds_data = {'Current Position Requirements': [],
              'Current Position': [], 'Alternative 1': [], 'Alternative 2': []}
y_preds_proba = logistic.predict_proba(X_test)

counter = 0
for idx, (pred_row, true_job_position) in enumerate(zip(y_preds_proba, y_test)):
    class_preds = np.argsort(pred_row)
    # delete true class
    for i in [-1, -2]:
        if class_preds[i] == true_job_position:
            class_preds = np.delete(class_preds, i)
    # getting other 2 highest job predictions
    top_classes = class_preds[-2:]
    # obtaining class name string from int label
    class_names = enc.inverse_transform(top_classes)
    true_job_position_name = enc.inverse_transform([true_job_position])
    # saving to dict
    preds_data['Current Position Requirements'].append(X_test_words.iloc[idx])
    preds_data['Current Position'].append(true_job_position_name[0])
    preds_data['Alternative 1'].append(class_names[1])
    preds_data['Alternative 2'].append(class_names[0])

    counter += 1

preds_df = pd.DataFrame.from_dict(preds_data)
preds_df.to_csv('Recommendations.csv', index=False)
preds_df


cm = confusion_matrix(y_test, test_preds)
plt.figure(figsize=(10, 10))
sns.heatmap(cm, annot=True, fmt="d")
plt.title("Confusion matrix")
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()


def make_prediction(text):
    text = [text]
    text = vectorizer.transform(text)
    text = text.toarray()
    pred = logistic.predict(text)
    return enc.inverse_transform(pred)[0]


# save model
pickle.dump(logistic, open('itreq.pkl', 'wb'))

preds_df

print(make_prediction(''' 
Proven working experience in project management.
Excellent client-facing and internal communication skills.
Excellent written and verbal communication skills.
Solid organizational skills including attention to detail and multitasking skills.
Strong working knowledge of Microsoft Office.
'''))
