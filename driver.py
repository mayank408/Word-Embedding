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
PATH = os.getcwd()
LOG_DIR = PATH+ '/embedding-log'

model = KeyedVectors.load_word2vec_format(filename, binary=True, limit=100000)
data_path = PATH + '/data'
data_dir_list = os.listdir(data_path)

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

    

    data_np = np.asarray(result, np.float32)

    num_of_samples=data_np.shape[0]
    num_of_samples_each_class = 100



    # features = tf.Variable(data_np, name='features')
    # print(features)

    y = np.ones((num_of_samples,),dtype='int64')

    y[0:100]=0
    y[100:200]=1
    y[200:300]=2
    y[300:]=3


    metadata_file = open(os.path.join(LOG_DIR, 'metadata_4_classes.tsv'), 'w')
    metadata_file.write('Class\tName\n')
    k=100 # num of samples in each class
    j=0
    #for i in range(210):
    #    metadata_file.write('%06d\t%s\n' % (i, names[y[i]]))
    for i in range(num_of_samples):
        c = word[y[i]]
        if i%k==0:
            j=j+1
        metadata_file.write('{}\t{}\n'.format(j,c))
        #metadata_file.write('%06d\t%s\n' % (j, c))
    metadata_file.close()
       
    

with tf.Session() as sess:
    saver = tf.train.Saver(result)

    sess.run(result.initializer)
    saver.save(sess, os.path.join(LOG_DIR, 'images_4_classes.ckpt'))
    
    config = projector.ProjectorConfig()
    # One can add multiple embeddings.
    embedding = config.embeddings.add()
    embedding.tensor_name = features.name
    # Link this tensor to its metadata file (e.g. labels).
    embedding.metadata_path = os.path.join(LOG_DIR, 'metadata_4_classes.tsv')
    # Comment out if you don't want sprites
    # embedding.sprite.image_path = os.path.join(LOG_DIR, 'sprite_4_classes.png')
    # embedding.sprite.single_image_dim.extend([img_data.shape[1], img_data.shape[1]])
    # Saves a config file that TensorBoard will read during startup.
    projector.visualize_embeddings(tf.summary.FileWriter(LOG_DIR), config)
    


    # pca = PCA(n_components=12)
    # result = pca.fit_transform(result)    

    # range_n_clusters = [2, 3, 5, 10, 20, 50, 100]       
    # best_clusters = 0                                   
    # previous_silh_avg = 0.0

    # print(result)

    # for n_clusters in range_n_clusters:
    #     if n_clusters < len(result):
    #         clusterer = KMeans(n_clusters=n_clusters)
    #         cluster_labels = clusterer.fit_predict(result)
    #         silhouette_avg = silhouette_score(result, cluster_labels)
    #         if silhouette_avg > previous_silh_avg:
    #             previous_silh_avg = silhouette_avg
    #             best_clusters = n_clusters


    # kmeans = KMeans(n_clusters=best_clusters, random_state=0).fit(result)
    # pyplot.scatter(result[:, 0], result[:, 1], c=kmeans.labels_)

    # i=0
    # for word in enumerate(word_dict):
    #     pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))
    #     i=i+1

    # # adjust_text(word, only_move='y', arrowprops=dict(arrowstyle="->", color='r', lw=0.5))

    # pyplot.show()

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