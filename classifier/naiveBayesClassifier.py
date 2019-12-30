"""Naive Bayes classifier."""
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

from classifier import Classifier
from commons.datamodel import DataModel


class NaiveBayesClassifier(Classifier):
    # Reads the data and stores in class variables
    def __init__(self):
        self.model = None
        self.training_data = None
        self.testing_data = None

    def trainModel(self, training_data: DataModel):
        count_vect = CountVectorizer(stop_words='english')
        X_train_counts = count_vect.fit_transform(training_data.documents)

        self.model = MultinomialNB()
        self.model.fit(X_train_counts, self.training_data.target)

    def classify(self, testing_data: DataModel):
        count_vect = CountVectorizer(stop_words='english')
        X_test_data = count_vect.transform(testing_data.documents)
        return self.model.predict(X_test_data)
        # score = metrics.accuracy_score(predict, self.testing_data.target)
        # print("Accuracy: {}".format(score))
