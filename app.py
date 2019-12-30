import asyncio
import json
import logging
import configparser
import time
from threading import Thread

import gensim
from flask import Flask           # import flask

from sklearn.datasets import fetch_20newsgroups

from classifier import Classifier
from classifier.sgdClassifier import SgdClassifier
from collectors.nyt_collector import NytCollector
from commons.datamodel import DataModel

app = Flask(__name__)             # create an app instance

logger = logging.getLogger(__name__)
CONFIG_FILE = "./config/prod.ini"
GLOVE_FILE = r'../glove.6B.300d.txt'
WORD_2_VEC_FILE = r'../glove.6B.300d_word2Vec.txt'
lake = [] # A list of data-models

CATEGORY_MAP = {
    'atheism': 0,
    'graphics': 1,
    'ms-windows-misc': 2,
    'ibm-hardware': 3,
    'mac-hardware': 4,
    'windows-x': 5,
    'forsale': 6,
    'autos': 7,
    'motorcycles': 8,
    'baseball': 9,
    'hockey': 10,
    'crypto': 11,
    'electronics': 12,
    'medicine': 13,
    'space': 14,
    'christianity': 15,
    'guns': 16,
    'mideast': 17,
    'politics': 18,
    'religion': 19
}

VOCABULARY = {
    'atheism': ["atheism"],
    'graphics': ["graphics"],
    'ms-windows-misc': ["windows", "microsoft"],
    'ibm-hardware': ["ibm", "hardware"],
    'mac-hardware': ["mac", "hardware"],
    'windows-x': ["windows"],
    'forsale': ["sale"],
    'autos': ["auto", "automobile"],
    'motorcycles': ["motorcyle", "bike"],
    'baseball': ["baseball"],
    'hockey': ["hockey"],
    'crypto': ["cryptography"],
    'electronics': ["electronics"],
    'medicine': ["medicine"],
    'space': ["space"],
    'christianity': ["christianity"],
    'guns': ["guns", "shooting", "lobby", "rifles", "weapon", "nra", "handgun", "politics"],
    'mideast': ["mideast"],
    'politics': ["politics"],
    'religion': ["religion"]
}


@app.route("/")                   # at the end point /
def home():
    resp = {'message': "Hello World!"}
    # show the complete stack rank for all categories
    return json.dumps(resp)


@app.route("/category/<name>")
def stackrank_by_category(name):
    resp = {'message': f"You chose {name} category"}
    # shows the stack rank for only the chosen category.
    return json.dumps(resp)


def train_classifier(classifier: Classifier):
    logger.info("Fetching 20Newsgroup data to train classifier")
    trainData_20newsgroup = fetch_20newsgroups(subset='train', shuffle=True)
    testData_20newsgroup = fetch_20newsgroups(subset='test', shuffle=True)

    testData = DataModel()
    testData.setData(testData_20newsgroup.data)
    testData.setTarget(testData_20newsgroup.target)

    classifier_training_data = DataModel()
    classifier_training_data.setData(trainData_20newsgroup.data)
    classifier_training_data.setTarget(trainData_20newsgroup.target)

    classifier.trainModel(classifier_training_data, testData)


def get_glove_model(gloveFile: str, word2VecFile: str):
    logger.info(f"Reading from Glove File: {gloveFile}. Converting to word2Vec format: {word2VecFile}")
    gensim.scripts.glove2word2vec.glove2word2vec(gloveFile, word2VecFile)
    model = gensim.models.KeyedVectors.load_word2vec_format(word2VecFile, binary=False)
    logger.info("Finishing gloVe learning.")
    return model


def build_stackrank(vocabulary):
    """Out of the complete set of predictions, given the vocabulary, build a top 10 for each category"""
    return


def enhance_vocabulary(glove_model, vocabulary):
    """Given the model, enhance the given vocabulary and return the enhanced vocabulary."""
    return vocabulary


if __name__ == "__main__":        # on running python app.py
    cparser = configparser.ConfigParser()
    cparser.read(CONFIG_FILE)
    logging.basicConfig(filename=cparser['Log']['filename'], level=cparser['Log']['level'])
    logger.info(f"Built config from {CONFIG_FILE}")

    nytCollector = NytCollector()
    nytCollector.build_config(config_file=CONFIG_FILE)

    # The event loop that runs all the collectors in parallel. TODO: refactor to handle multiple collectors
    collector_loop = asyncio.new_event_loop()
    t = Thread(target=nytCollector.run, args=(collector_loop,))
    t.start()

    logger.info("Training classifier.")
    classifier = SgdClassifier()
    train_classifier(classifier)

    while not nytCollector.data:
        logger.info("Collector has not finished collecting data yet, sleeping for 60 seconds.")
        time.sleep(60)

    # TODO: refactor to handle multiple collectors.
    lake.append(nytCollector.data)
    predictions = []
    for datamodel in lake:
        classifier.classify(datamodel)

    model = get_glove_model(GLOVE_FILE, WORD_2_VEC_FILE)
    vocab = enhance_vocabulary(glove_model=model, vocabulary=VOCABULARY)
    build_stackrank(vocabulary=vocab)
    app.run(debug=True)                     # run the flask app

