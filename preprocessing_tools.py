import re
import html
import string
import spacy
from spacy.lang.en.stop_words import STOP_WORDS


punctuations = string.punctuation
nlp = spacy.load("en_core_web_sm")
stop_words = spacy.lang.en.stop_words.STOP_WORDS


def text_cleaner(sentence):
    # unescape
    sentence = html.unescape(sentence)

    # remove HTML tags
    regex = re.compile('<.*?>')
    sentence = regex.sub('', sentence)

    # remove punctuation
    regex = re.compile('[%s]' % re.escape(string.punctuation))
    sentence = regex.sub('', sentence)

    # tokenize and lemmatize
    sentence = ' '.join([word.lemma_ for word in nlp(sentence)])

    # remove stop words
    regex = re.compile(' | '.join(stop_words))
    sentence = regex.sub(' ', sentence)

    regex = re.compile(r'\s+')
    sentence = regex.sub(' ', sentence)

    return sentence.split(' ')