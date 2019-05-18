import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
from nltk.stem import WordNetLemmatizer
import nltk
import string


def draw_freq_graph(word_list, title):
    word_dist = nltk.FreqDist(word_list)
    result = pd.DataFrame(word_dist.most_common(25), columns=['Word', 'Frequency']).set_index('Word')

    matplotlib.style.use('ggplot')
    result.plot.bar(rot=0)
    plt.title(title)
    plt.show()


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

draw_freq_graph(all_words, 'All Words')

sarcastic = data[data.is_sarcastic == 1]
factual = data[data.is_sarcastic == 0]

sar_headlines = sarcastic.apply(lambda row: tokenizer.tokenize(row['headline']), axis=1)
fact_headlines = factual.apply(lambda row: tokenizer.tokenize(row['headline']), axis=1)

# List of words in each category's headline that is not punctuation or a stopword
sar_words = [word for headline in sar_headlines for word in headline if word not in stops and word not in string.punctuation]
draw_freq_graph(sar_words, 'Sarcastic Words')

fact_words = [word for headline in fact_headlines for word in headline if word not in stops and word not in string.punctuation]
draw_freq_graph(fact_words, 'Factual Words')

lemmatizer = WordNetLemmatizer()
lem_sar_words = [lemmatizer.lemmatize(word) for word in sar_words]
lem_sar_freq = nltk.FreqDist(lem_sar_words)
print(lem_sar_freq.most_common(25))
draw_freq_graph(lem_sar_words, 'Lemmatized Sarcastic Words')
