import joblib
import pandas as pd
from nltk.corpus import stopwords
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import train_test_split
# import nltk
# nltk.download('stopwords')

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

vectorizer = TfidfVectorizer(max_features=5000, min_df=7, max_df=0.8, stop_words=stopwords.words('english'))
processed_features = vectorizer.fit_transform(data['Text']).toarray()

x, x_test, y, y_test = train_test_split(processed_features, data['Alignment'], test_size=0.2, random_state=0)

# Random Forest
text_classifier = RandomForestClassifier(n_estimators=100, random_state=0)
text_classifier.fit(x, y)
predictions = text_classifier.predict(x_test)
print(predictions)

print(text_classifier.score(x_test, y_test))
print(text_classifier.predict(vectorizer.transform(['Trump is a genius'])))

# save the model
joblib.dump(text_classifier, "model.pkl")
# load the model
model = joblib.load("model.pkl")

# print(confusion_matrix(y_test, predictions))
# print(classification_report(y_test, predictions))
# print(accuracy_score(y_test, predictions))
