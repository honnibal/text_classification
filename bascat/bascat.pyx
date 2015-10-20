from __future__ import division

from thinc.features cimport Feature
from preshed.counter cimport PreshCounter, count_t
from preshed.maps cimport PreshMap

from spacy.attrs cimport ORTH, LEMMA, IS_PUNCT, CLUSTER, IS_ALPHA
from spacy.typedefs cimport attr_t
from spacy.tokens.doc cimport Doc
from spacy.tokens.token cimport Token
from spacy.lexeme cimport Lexeme

from spacy.structs cimport TokenC

from thinc.features cimport count_feats
from thinc.learner cimport LinearModel
from cymem.cymem cimport Pool, Address

from libc.stdint cimport int64_t, uint64_t, uint32_t


cdef extern from "stdlib.h":
    double drand48() nogil
    void srand48(long int seedval) nogil


srand48(100)


cdef int arg_max(const float* scores, const int n_classes) nogil:
    cdef int i
    cdef int best = 0
    cdef float mode = scores[0]
    for i in range(1, n_classes):
        if scores[i] > mode:
            mode = scores[i]
            best = i
    return best


cdef class BagOfWords:
    cdef Pool mem
    cdef Feature* c
    cdef int length
    cdef int capacity

    @staticmethod
    cdef int reject(const TokenC* token, float dropout) nogil:
        if not Lexeme.c_check_flag(token.lex, IS_ALPHA):
            return True
        elif token.ent_iob == 1 or token.ent_iob == 3:
            return True
        elif dropout and drand48() < dropout:
            return True
        else:
            return False

    @classmethod
    def from_doc(cls, Doc doc, float dropout):
        cdef PreshMap bag = PreshMap()
        cdef int i
        cdef attr_t key
        cdef attr_t vocab_offset = doc.vocab.strings.size
        cdef uint64_t bigram = 0
        cdef uint32_t part1 = 0
        cdef uint32_t part2 = 1
        for i in range(doc.length):
            if not BagOfWords.reject(&doc.data[i], dropout):
                key = doc.data[i].lex.orth
                bag.set(key, <void*>(<count_t>bag.get(key) + 1))
                key = doc.data[i].lex.cluster + vocab_offset
                bag.set(key, <void*>(<count_t>bag.get(key) + 1))
                part1 = doc.data[i].lex.orth
                part2 = doc.data[i + doc.data[i].head].lex.orth
                bigram = <uint64_t>part1 << 32 | part2
                bag.set(bigram, <void*>(<count_t>bag.get(key) + 1))
        return cls(bag.items())

    def __init__(self, counts=None):
        self.mem = Pool()
        self.length = 0
        self.capacity = 8
        self.c = <Feature*>self.mem.alloc(self.capacity, sizeof(Feature))
        cdef uint64_t orth
        cdef int count
        if counts is not None:
            for orth, count in counts:
                self.add(orth, count)

    cdef int add(self, uint64_t orth, float count) nogil:
        if self.length == self.capacity:
            self.capacity *= 2
            with gil:
                self.c = <Feature*>self.mem.realloc(self.c, self.capacity * sizeof(Feature))
        self.c[self.length].i = 0
        self.c[self.length].key = orth
        self.c[self.length].value = count
        self.length += 1


cdef class TextClassifier:
    cdef int n_classes
    cdef LinearModel model

    @classmethod
    def train(cls, instances, n_classes, float dropout, n_iter, callback=None):
        cdef TextClassifier self = cls(n_classes, model_dir=None)
        for i in range(n_iter):
            loss = 0
            for doc, gold_class in instances:
                feats = self.extract(doc, dropout)
                guess_class = self.predict(feats)
                self.update(feats, guess_class, gold_class, 1)
                loss += guess_class != gold_class
            print(1 - (loss / len(instances)))
        self.model.end_training()
        return self

    def __init__(self, n_classes, model_dir=None):
        self.n_classes = n_classes
        self.model = LinearModel(self.n_classes, 2)

    def extract(self, Doc doc, float dropout):
        return BagOfWords.from_doc(doc, dropout)

    def predict(self, BagOfWords bow):
        cdef Address scores_addr = Address(self.n_classes, sizeof(float))
        scores = <float*>scores_addr.ptr
        self.model.set_scores(scores, bow.c, bow.length)
        return arg_max(scores, self.n_classes)

    def update(self, BagOfWords bow, int guess, int gold, float cost):
        counts = {gold: {}, guess: {}}
        count_feats(counts[gold], bow.c, bow.length, cost)
        count_feats(counts[guess], bow.c, bow.length, -cost)
        self.model.update(counts)
