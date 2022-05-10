from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd

sentimentAnalyser = SentimentIntensityAnalyzer()

data = pd.read_csv('all_articles_cleaned.csv')
data = data.drop(data.columns[0], axis=1)
data = data.drop(labels=['Title', 'URL'], axis=1)
print(data.head())


def calculate_sentiment(text):
    # Run VADER on the text
    scores = sentimentAnalyser.polarity_scores(text)
    # Extract the compound score
    compound_score = scores['compound']
    # Return compound score
    return compound_score


# Apply the function to every row in the "text" column and output the results into a new column "sentiment_score"
data['sentiment_score'] = data['Text'].apply(calculate_sentiment)
print(data.sort_values(by='sentiment_score', ascending=False)[:10])
data.to_csv('vader.csv')

#calculate_sentiment('I like the Marvel movies')

# # calculate the negative, positive, neutral and compound scores, plus verbal evaluation
# def sentiment_vader(sentence):
#     # Create a SentimentIntensityAnalyzer object.
#     sid_obj = SentimentIntensityAnalyzer()
#
#     sentiment_dict = sid_obj.polarity_scores(sentence)
#     negative = sentiment_dict['neg']
#     neutral = sentiment_dict['neu']
#     positive = sentiment_dict['pos']
#     compound = sentiment_dict['compound']
#
#     if sentiment_dict['compound'] >= 0.05:
#         overall_sentiment = "Positive"
#
#     elif sentiment_dict['compound'] <= - 0.05:
#         overall_sentiment = "Negative"
#
#     else:
#         overall_sentiment = "Neutral"
#
#     #print(negative)
#     #print(neutral)
#     #print(positive)
#     print("Hello!", compound)
#     #print(overall_sentiment)
#     return negative, neutral, positive, compound, overall_sentiment
#
#
# if __name__ == "__main__":
#
#     file_name = 'output.txt'
#     text_file = open('output.txt', 'r')
#     sentences = text_file.readlines()
#     count = 1
#     for sentence in sentences:
#         #print(sentence)
#         count = count + 1
#         sentiment_vader(sentence)
#         if count > 2:
#              break