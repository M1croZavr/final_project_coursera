from sentiment_classifier import SentimentClassifier

clf = SentimentClassifier()

pred = clf.get_prediction_message("It's a great banking, love it")

print(pred)
