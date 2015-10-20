from __future__ import division

import plac
from pathlib import Path
import random

import spacy.en

from bascat.bascat import BagOfWords, TextClassifier


def read_data(nlp, data_dir):
    for subdir, label in (('pos', 1), ('neg', 0)):
        for filename in (data_dir / subdir).iterdir():
            text = filename.open().read()
            doc = nlp(text)
            if len(doc) >= 1:
                yield doc, label


def partition(examples, split_size):
    examples = list(examples)
    random.shuffle(examples)
    n_docs = len(examples)
    split = int(n_docs * split_size)
    return examples[:split], examples[split:]


def iter_data(examples, n_iter):
    for _ in range(n_iter):
        for doc, label in examples:
            yield doc, label


@plac.annotations(
    data_dir=("Data directory", "positional", None, Path),
    n_iter=("Number of iterations (epochs)", "option", "i", int),
    dropout=("Drop-out rate", "option", "r", float),
)
def main(data_dir, n_iter=5, dropout=0.3):
    n_classes = 2
    print("Loading")
    nlp = spacy.en.English()
    print("Processing docs")
    train_data, dev_data = partition(read_data(nlp, data_dir / 'train'), 0.8)
    eval_data = list(read_data(nlp, data_dir / 'test'))
    print("Train")
    print(len(train_data))
    model = TextClassifier.train(train_data, n_classes, dropout, n_iter)

    print("Evaluating")
    n_correct = 0
    for x, y in eval_data:
        guess = model.predict(model.extract(x, 0.0))
        n_correct += guess == y
    print(n_correct / len(eval_data))
 

if __name__ == '__main__':
    plac.call(main)
