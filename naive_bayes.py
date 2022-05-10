import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

data = pd.read_csv('all_articles_cleaned.csv')
data = data.drop(data.columns[0], axis=1)
data = data.drop(labels=['Title', 'URL'], axis=1)
print(data.head())

data['Alignment'] = data['Alignment'].replace({
    'right': 0,
    'right-center': 1,
    'center': 2,
    'left-center': 3,
    'left': 4
})

# Split into training and testing data
x = data['Text']
y = data['Alignment']
x, x_test, y, y_test = train_test_split(x,y, stratify=y, test_size=0.25, random_state=42)


# Vectorize text to numbers
vec = CountVectorizer(stop_words='english')
x = vec.fit_transform(x).toarray()
x_test = vec.transform(x_test).toarray()

model = MultinomialNB()
model.fit(x, y)

# Check the correctness of the model
print(model.score(x_test, y_test))
print(model.predict(vec.transform(['Gun violence is bad'])))

