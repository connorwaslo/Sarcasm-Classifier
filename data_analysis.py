import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
import nltk
import string

pd.set_option('display.max_columns', 5)

data = pd.read_json('Sarcasm_Headlines_Dataset.json', lines=True)
category_counts = data['is_sarcastic'].value_counts()

# Visualize data split
# plt.pie(category_counts, labels=['Not Sarcastic', 'Sarcastic'])
# plt.show()

tokenizer = TweetTokenizer()
headlines = data.apply(lambda row: tokenizer.tokenize(row['headline']), axis=1)
# print(headlines)

all_words = []
stops = set(stopwords.words('english'))
for headline in headlines:
    for word in headline:
        if word not in stops and word not in string.punctuation:
            all_words.append(word)

word_dist = nltk.FreqDist(all_words)

result = pd.DataFrame(word_dist.most_common(25), columns=['Word', 'Frequency']).set_index('Word')
print(result)

matplotlib.style.use('ggplot')
result.plot.bar(rot=0)
plt.show()
