import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_json('Sarcasm_Headlines_Dataset.json', lines=True)

category_counts = data['is_sarcastic'].value_counts()

# Data split
plt.pie(category_counts, labels=['Not Sarcastic', 'Sarcastic'])
plt.show()
