#!/usr/bin/env python
__author__ = "Prashant Shivarm Bhat"
__email__ = "PrashantShivaram@outlook.com"

import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords

nltk.download('stopwords')


def tokenize_words(inp):
    inp = inp.lower()
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(inp)
    filtered = filter(lambda token: token not in stopwords.words('english'), tokens)
    return " ".join(filtered)
