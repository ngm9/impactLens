"""The classifier class encodes all kinds of classifiers we experiment with. This class is used as a super class for
all classifiers."""

from commons.datamodel import DataModel


class Classifier:

    def trainModel(self, training_data: DataModel):
        pass

    def classify(self, testing_data: DataModel):
        pass

