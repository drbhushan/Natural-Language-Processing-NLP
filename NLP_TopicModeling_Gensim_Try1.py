# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 14:30:11 2019

@author: bbhushan
"""

doc1 = "Cricket is a bat-and-ball game."
doc2 = "A car or automobile is a wheeled motor vehicle used for transportation"
doc3 = "Cricket is played between two teams of eleven players on a field at the centre of which is a 20-metre"
doc4 = "Cars have controls for driving, parking, passenger comfort, and a variety of lights."
doc5 = "Cricket is a bat-and-ball game. A car or automobile is a wheeled motor vehicle used for transportation"

doc_complete = [doc1, doc2, doc3, doc4, doc5]


from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string
stop = set(stopwords.words('english'))
exclude = set(string.punctuation)
lemma = WordNetLemmatizer()

def clean(doc):
    stop_free = ' '.join([i for i in doc.lower().split() if i not in stop])
    punc_free = ''.join([ch for ch in stop_free if ch not in exclude])
    normalized = ' '.join(lemma.lemmatize(word) for word in punc_free.split())
    
    return normalized
doc_clean = [clean(doc).split() for doc in doc_complete]

import gensim
from gensim import corpora
dictionary = corpora.Dictionary(doc_clean)
doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]


Lda = gensim.models.ldamodel.LdaModel
ldamodel = Lda(doc_term_matrix, num_topics = 3, id2word = dictionary, passes=50)

print(ldamodel.print_topics(num_topics=3, num_words=4))