from sentiment_classifier import SentimentClassifier


clf = SentimentClassifier()
pred = clf.get_prediction_message('Отличный смартфон, очень хорошая камера и быстрый процессор')
print(pred)
