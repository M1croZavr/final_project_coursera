import joblib


class SentimentClassifier(object):
    """
    A class used to represent a classifier pipeline

    ...

    Attributes
    ----------
    pipeline: sklearn.pipeline.Pipeline
        a pipeline used to transform text and make predictions
    classes_dict: dict
        dictionary represents class number and its name

    Methods
    -------
    get_probability_words(probability): staticmethod
        returns degree of certainty according to probability
    predict_text(text)
        returns predicted class and probability of this class
    predict_list(list_of_texts)
        returns predicted class and probability of this class for each text in list
    get_prediction_message(text)
        returns generated text according to probability and class
    """
    def __init__(self):
        self.pipeline = joblib.load("best_pipeline_0944.pkl")
        self.classes_dict = {0: 'negative',
                             1: 'positive',
                            -1: 'prediction error'}

    @staticmethod
    def get_probability_words(probability):
        if probability < 0.55:
            return 'neutral or uncertain'
        if probability < 0.75:
            return 'probably'
        if probability > 0.90:
            return 'certain'
        else:
            return ''

    def predict_text(self, text):
        try:
            return self.pipeline.predict([text])[0],\
                   self.pipeline.predict_proba([text])[0].max()
        except:
            print('prediction error')
            return -1, 0.8

    def predict_list(self, list_of_texts):
        try:
            return self.pipeline.predict(list_of_texts),\
                   self.pipeline.predict_proba(list_of_texts)
        except:
            print('prediction error')
            return None

    def get_prediction_message(self, text):
        class_prediction, prediction_probability = self.predict_text(text)
        return self.get_probability_words(prediction_probability) + ' ' + self.classes_dict[class_prediction]
