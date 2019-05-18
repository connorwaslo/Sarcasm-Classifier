import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import nltk

pd.set_option('display.max_columns', 5)

data = pd.read_json('Sarcasm_Headlines_Dataset.json', lines=True)
print(data)

category_counts = data['is_sarcastic'].value_counts()

# Data split
plt.pie(category_counts, labels=['Not Sarcastic', 'Sarcastic'])
plt.show()

headlines = data['headline'].str.lower().str.replace(r'\|', ' ').str.cat(sep=' ')
words = nltk.tokenize.word_tokenize(headlines)
word_dist = nltk.FreqDist(words)

result = pd.DataFrame(word_dist.most_common(25), columns=['Word', 'Frequency']).set_index('Word')
print(result)

matplotlib.style.use('ggplot')
result.plot.bar(rot=0)
plt.show()
