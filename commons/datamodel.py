"""The data model encodes any data set that we are using, either for training or learning."""


class DataModel:
    documents = []  # A list of documents.
    target = []  # A corresponding list of classification if supervised learning requires.
    scores = {}  # A stack rank of articles in the descending order of their impact scores.
    metadata = []  # A list of metadata for each of the documents. e.g. URLs
    id = []  # A list of identification metrics for each document.

    def __init__(self):
        self.documents = None
        self.target = None
        self.scores = None
        self.metadata = None
        self.id = None

    def setDocuments(self, data):
        self.documents = data

    def setTarget(self, target):
        self.target = target
