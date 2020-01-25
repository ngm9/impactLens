"""The data model encodes any data set that we are using, either for training or learning."""
from collections import defaultdict

from commons.articlemodel import ArticleModel


class DataModel:
    documents = []  # A list of documents with any datamodel example: ArticleModel
    # TODO: convert these dictionaries into lists - save redundant use of article data object as keys.
    targetCategoryList = []  # (article, category) tuple list for each of the Documents above. Frivolous use of data, but well..
    scores = []  # (article, score) tuple of each article from Documents. Frivolous use of data, but well..

    def setDocumentsFromRawTextArray(self, textArray):
        self.documents = [ArticleModel(text=text, url="", headline="") for i, text in enumerate(textArray)]

    def setDocumentsFromDataModelArray(self, dataArray):
        self.documents = dataArray

    def setTargetCategories(self, target):
        self.targetCategoryList = target

