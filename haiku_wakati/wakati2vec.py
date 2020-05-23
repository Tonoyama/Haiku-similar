# -*- coding: utf-8 -*-

from gensim.models import word2vec

sentences = word2vec.Text8Corpus('./haiku_wakati.txt')

model = word2vec.Word2Vec(sentences, sg=1, size=200, min_count=1, window=15) 
model.save("./haiku_wakati.model")
