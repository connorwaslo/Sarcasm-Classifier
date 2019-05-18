import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
from nltk import FreqDist
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.ensemble import VotingClassifier

from sklearn import metrics
import numpy as np
import string

data = pd.read_json('Sarcasm_Headlines_Dataset.json', lines=True)

tokenizer = TweetTokenizer()
headlines = data.apply(lambda row: tokenizer.tokenize(row['headline']), axis=1)

# Set of all stopwords
stops = set(stopwords.words('english'))

# Frequency of all words
all_words = [word for headline in headlines for word in headline if word not in stops and word not in string.punctuation]
word_dist = FreqDist(all_words)

# features = []
# for headline in headlines:
#     line = []
#     for word in headline:
#         if word not in stops and word not in string.punctuation:
#             line.append(word)
#     features.append(line)

word_features = list(word_dist.keys())[:8000]  # Include most common 3000 words as features


# Bag of words method of creating a vector
def feature_vec(headline):
    words = set(headline)
    dict_feats = {}
    for w in word_features:
        dict_feats[w] = (w in words)

    feats = []
    for key in dict_feats.keys():
        if dict_feats[key]:
            feats.append(1)
        else:
            feats.append(0)

    return feats


features = []
for headline in headlines:
    features.append(feature_vec(headline))

print(len(features), np.array(features).shape)

labels = data['is_sarcastic'].tolist()
train_x, val_x, train_y, val_y = train_test_split(features, labels, test_size=0.2)

print(len(train_x), len(train_y))
print(len(val_x), len(val_y))

gnb = GaussianNB()
gnb.fit(train_x, train_y)
gnb_pred_y = gnb.predict(val_x)
print('GaussianNB Accuracy:', metrics.accuracy_score(val_y, gnb_pred_y))

bern = BernoulliNB()
bern.fit(train_x, train_y)
bern_pred_y = bern.predict(val_x)
print('BernoulliNB Accuracy:', metrics.accuracy_score(val_y, bern_pred_y))

multi = MultinomialNB()
multi.fit(train_x, train_y)
multi_pred_y = multi.predict(val_x)
print('MultinomialNB Accuracy:', metrics.accuracy_score(val_y, multi_pred_y))

log = LogisticRegression()
log.fit(train_x, train_y)
log_pred_y = log.predict(val_x)
print('Logistic Regression Accuracy:', metrics.accuracy_score(val_y, log_pred_y))

sgd = SGDClassifier()
sgd.fit(train_x, train_y)
sgd_pred_y = sgd.predict(val_x)
print('SGDClassifier Accuracy:', metrics.accuracy_score(val_y, sgd_pred_y))

vote = VotingClassifier([('gnb', gnb), ('bern', bern), ('multi', multi), ('lr', log), ('sgd', sgd)])
vote.fit(train_x, train_y)
vote_pred_y = vote.predict(val_x)
print('Vote Accuracy:', metrics.accuracy_score(val_y, vote_pred_y))
