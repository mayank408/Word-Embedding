import logging
import os
from sklearn.decomposition import PCA
from gensim.models import KeyedVectors
import numpy as np
from matplotlib import pyplot
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from adjustText import adjust_text
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector

filename = 'GoogleNews-vectors-negative300.bin'

model = KeyedVectors.load_word2vec_format(filename, binary=True, limit=100000)

def read_directory(mypath):
    current_list_of_files = []

    while True:
        for (_, _, filenames) in os.walk(mypath):
            current_list_of_files = filenames
        logging.info("Reading the directory for the list of file names")
        return current_list_of_files


def creating_subclusters(list_of_terms, name_of_file, result, word_dict):
    
    not_in_vocab = []

    for word in list_of_terms:
        try:
            res = model.word_vec(word)
            result.append(res)
            word_dict.append(word)
        except:
            not_in_vocab.append(word)

    pca = PCA(n_components=12)
    result = pca.fit_transform(result)    

    range_n_clusters = [2, 3, 5, 10, 20, 50, 100]       
    best_clusters = 0                                   
    previous_silh_avg = 0.0

    print(result)

    for n_clusters in range_n_clusters:
        if n_clusters < len(result):
            clusterer = KMeans(n_clusters=n_clusters)
            cluster_labels = clusterer.fit_predict(result)
            silhouette_avg = silhouette_score(result, cluster_labels)
            if silhouette_avg > previous_silh_avg:
                previous_silh_avg = silhouette_avg
                best_clusters = n_clusters


    kmeans = KMeans(n_clusters=best_clusters, random_state=0).fit(result)
    pyplot.scatter(result[:, 0], result[:, 1], c=kmeans.labels_)

    i=0
    for word in enumerate(word_dict):
        pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))
        i=i+1

    # adjust_text(word, only_move='y', arrowprops=dict(arrowstyle="->", color='r', lw=0.5))

    pyplot.show()

    pass





# Main function
if __name__ == '__main__':
    result = []
    word_dict = []
    logging.basicConfig(level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')

    mypath = "SampleExamples/input"
    list_of_input_files = read_directory(mypath)
    for each_file in list_of_input_files:
        with open(os.path.join(mypath, each_file), "r") as f:
            file_contents = f.read()
        list_of_term_in_cluster = file_contents.split()

        creating_subclusters(list_of_term_in_cluster, each_file, result, word_dict)


        # End of code