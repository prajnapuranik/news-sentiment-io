from textblob import TextBlob

import requests
from bs4 import BeautifulSoup
import pandas as pd
import spacy
nlp = spacy.load("en_core_web_sm")


# data scraping
r = requests.get('https://www.newsy.com/stories/commercial-companies-advance-space-exploration/')
r.encoding = 'utf-8'
html = r.text
soup = BeautifulSoup(html, features="lxml")
text = soup.get_text()
len(text)

clean_text = text.replace("\n", " ")
clean_text = clean_text.replace("/", " ")
clean_text = ''.join([c for c in clean_text if c != "\'"])

sentence = []
tokens = nlp(clean_text)
for sent in tokens.sents:
    sentence.append((sent.text.strip()))

#sentences are not ready to be analyzed
textblob_sentiment =[]
for s in sentence:
    txt = TextBlob(s)
    a = txt.sentiment.polarity
    b = txt.sentiment.subjectivity
    textblob_sentiment.append([s,a,b])

df_textblob = pd.DataFrame(textblob_sentiment, columns = ['Sentence', 'Polarity', 'Subjectivity'])
# print(df_textblob.head())
print(df_textblob.to_csv)

html = df_textblob.to_html(index=False)
text_data = df_textblob.to_csv(sep="/", index=False)

# write html to file
text_file = open("index.html", "w")
text_file2 = open("output.txt", "w")
text_file.write(html)
text_file2.write(text_data)
text_file.close()
text_file2.close()

