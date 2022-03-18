import joblib


class SentimentClassifier(object):
    def __init__(self):
        self.pipeline = joblib.load("pipeline_from_kaggle.pkl")
        self.classes_dict = {0: "negative",
                             1: "positive",
                            -1: "prediction error"}

    @staticmethod
    def get_probability_words(probability):
        if probability < 0.55:
            return "neutral or uncertain"
        if probability < 0.7:
            return "probably"
        if probability > 0.95:
            return "certain"
        else:
            return ""

    def predict_text(self, text):
        try:
            return self.pipeline.predict([text])[0],\
                   self.pipeline.predict_proba([text])[0].max()
        except:
            print("prediction error")
            return -1, 0.8

    def predict_list(self, list_of_texts):
        try:
            return self.pipeline.predict(list_of_texts),\
                   self.pipeline.predict_proba(list_of_texts)
        except:
            print('prediction error')
            return None

    def get_prediction_message(self, text):
        class_prediction, prediction_probability = prediction = self.predict_text(text)
        return self.get_probability_words(prediction_probability) + " " + self.classes_dict[class_prediction]
