# Just disables the warning, doesn't enable AVX/FMA
import os,cv2
import numpy as np
import matplotlib.pyplot as plt
import pickle
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
tf.__version__

PATH = os.getcwd()

LOG_DIR = PATH+ '/embedding-logs'
#metadata = os.path.join(LOG_DIR, 'metadata2.tsv')

#%%
data_path = PATH + '/data'
data_dir_list = os.listdir(data_path)

img_data=[]
for dataset in data_dir_list:
    img_list=os.listdir(data_path+'/'+ dataset)
    print ('Loaded the images of dataset-'+'{}\n'.format(dataset))
    for img in img_list:
        input_img=cv2.imread(data_path + '/'+ dataset + '/'+ img )
        input_img_resize=cv2.resize(input_img,(224,224))
        img_data.append(input_img_resize)
    
                
img_data = np.array(img_data)

#%%

feature_vectors = np.loadtxt('feature_vectors_400_samples.txt')
print ("feature_vectors_shape:",feature_vectors.shape)
print ("num of images:",feature_vectors.shape[0])
print ("size of individual feature vector:",feature_vectors.shape[1])

num_of_samples=feature_vectors.shape[0]
num_of_samples_each_class = 100

features = tf.Variable(feature_vectors, name='features')


y = np.ones((num_of_samples,),dtype='int64')

y[0:100]=0
y[100:200]=1
y[200:300]=2
y[300:]=3

names = ['cats','dogs','horses','humans']

#with open(metadata, 'w') as metadata_file:
#    for row in range(210):
#        c = y[row]
#        metadata_file.write('{}\n'.format(c))
metadata_file = open(os.path.join(LOG_DIR, 'metadata_4_classes.tsv'), 'w')
metadata_file.write('Class\tName\n')
k=100 # num of samples in each class
j=0
#for i in range(210):
#    metadata_file.write('%06d\t%s\n' % (i, names[y[i]]))
for i in range(num_of_samples):
    c = names[y[i]]
    if i%k==0:
        j=j+1
    metadata_file.write('{}\t{}\n'.format(j,c))
    #metadata_file.write('%06d\t%s\n' % (j, c))
metadata_file.close()
       
    
# Taken from: https://github.com/tensorflow/tensorflow/issues/6322

#%%
with tf.Session() as sess:
    saver = tf.train.Saver([features])

    sess.run(features.initializer)
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
    

    