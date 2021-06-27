import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from time import time
import json

ted_main_df = pd.read_csv(r"D:\lucru\Licenta Valencia based\Git repo\A-Semi-supervised-Method-to-Classify-Educational-Videos\original_dataset\ted_main.csv", encoding='utf-8')

import ast
ted_main_df['tags'] = ted_main_df['tags'].apply(lambda x: ast.literal_eval(x))
s = ted_main_df.apply(lambda x: pd.Series(x['tags']),axis=1).stack().reset_index(level=1, drop=True)
s.name = 'theme'
theme_df = ted_main_df.drop('tags', axis=1).join(s)
theme_df.head()
len(theme_df['theme'].value_counts())
pop_themes = pd.DataFrame(theme_df['theme'].value_counts()).reset_index()
pop_themes.columns = ['theme', 'talks']
pop_themes.head(10)
plt.figure(figsize=(15,5))
sns.barplot(x='theme', y='talks', data=pop_themes.head(10))
#plt.show()

with open(r'D:\lucru\Licenta Valencia based\Git repo\A-Semi-supervised-Method-to-Classify-Educational-Videos\original_dataset\transcripts.json',"r", encoding='utf-8') as f:
    videos_json = json.load(f)

transcripts = []

for video in videos_json:
    url = video["url"]
    transcript = video["transcript"]
    if len(transcript) > 100 and len(transcript) < 1000:
        a = 1
    else:
        transcripts.append(len(transcript))

print(len(transcripts))
plt.hist(transcripts,density=0, bins=40)
plt.ylabel('Transcripts')
plt.xlabel('Characters')
plt.show()