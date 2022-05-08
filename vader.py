from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


# calculate the negative, positive, neutral and compound scores, plus verbal evaluation
def sentiment_vader(sentence):
    # Create a SentimentIntensityAnalyzer object.
    sid_obj = SentimentIntensityAnalyzer()

    sentiment_dict = sid_obj.polarity_scores(sentence)
    negative = sentiment_dict['neg']
    neutral = sentiment_dict['neu']
    positive = sentiment_dict['pos']
    compound = sentiment_dict['compound']

    if sentiment_dict['compound'] >= 0.05:
        overall_sentiment = "Positive"

    elif sentiment_dict['compound'] <= - 0.05:
        overall_sentiment = "Negative"

    else:
        overall_sentiment = "Neutral"

    #print(negative)
    #print(neutral)
    #print(positive)
    print("Hello!", compound)
    #print(overall_sentiment)
    return negative, neutral, positive, compound, overall_sentiment


if __name__ == "__main__":

    file_name = 'output.txt'
    text_file = open('output.txt', 'r')
    sentences = text_file.readlines()
    count = 1
    for sentence in sentences:
        #print(sentence)
        count = count + 1
        sentiment_vader(sentence)
        if count > 2:
             break

