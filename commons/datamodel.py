"""The data model encodes any data set that we are using, either for training or learning."""


class DataModel:
    documents = []  # A list of documents.
    target = []  # A corresponding list of classification if supervised learning requires.

    def __init__(self):
        self.documents = None
        self.target = None

    def setData(self, data):
        self.documents = data

    def setTarget(self, target):
        self.target = target
