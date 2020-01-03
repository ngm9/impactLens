"""The data model encodes any data set that we are using, either for training or learning."""
from collections import defaultdict


class DataModel:
    documents = []  # A list of documents.
    target = []  # A corresponding list of classification if supervised learning requires.
    scores = defaultdict()  # A stack rank of articles in the descending order of their impact scores.
    id = []  # A list of id for each of the documents. e.g. URLs
    meta = []  # A list of metadata metrics for each document. e.g. Headlines

    def setDocuments(self, data):
        self.documents = data

    def setTarget(self, target):
        self.target = target
