# -*- coding: utf-8 -*-
from sklearn.decomposition import PCA
from gensim.models import KeyedVectors
import numpy as np
from matplotlib import pyplot
# load the google word2vec model
filename = 'GoogleNews-vectors-negative300.bin'


model = KeyedVectors.load_word2vec_format(filename, binary=True, limit=50000)
# calculate: (king - man) + woman = ?
result = model.word_vec('china')

result = np.reshape(result, (1, -1))



pca = PCA(n_components=2)
result = pca.fit_transform(result)
print(result)


# pyplot.scatter(result[:, 0], result[:, 1])
# words = list(result)
# for i, word in enumerate(words):
# 	pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))
# pyplot.show()


