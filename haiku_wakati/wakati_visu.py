# -*- coding: utf-8 -*-

import gensim
import torch
from tensorboardX import SummaryWriter

vec_path = "./haiku_wakati.model"


writer = SummaryWriter()
# model = gensim.models.KeyedVectors.load_word2vec_format(vec_path, binary=True)
model = gensim.models.Word2Vec.load('./haiku_wakati.model')
weights = model.wv.vectors
labels = model.wv.index2word

# DEBUG: visualize vectors up to 1000
weights = weights[:3000]
labels = labels[:3000]

writer.add_embedding(torch.FloatTensor(weights), metadata=labels)
