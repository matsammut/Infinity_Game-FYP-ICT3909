import os
import cv2
import numpy as np
import sklearn.cluster
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import face_recognition
import shutil
from BestFace import *

def DBSCAN():
    # Load the images from a folder
    folder_path = 'Film/Test'
    images = []
    image_filenames = []
    for filename in os.listdir(folder_path):
        # print(filename)
        img = face_recognition.load_image_file(os.path.join(folder_path, filename))
        images.append(img)
        image_filenames.append(filename)

    # Detect and extract face embeddings from each image
    embeddings = []
    for img in images:
        face_location = (0, img.shape[0], img.shape[1], 0)
        face_encodings = face_recognition.face_encodings(img, [face_location])
        embeddings.append(face_encodings[0])

    # Cluster the embeddings using DBSCAN algorithm
    dbscan = sklearn.cluster.DBSCAN(eps=0.35, min_samples=15)
    labels = dbscan.fit_predict(embeddings)

    # Group the clustered images into an array of arrays with the elements being the image file names
    clustered_images = []
    for label in np.unique(labels):
        indices = np.where(labels == label)[0]
        clustered_images.append([image_filenames[i] for i in indices])

    # Plot the clustered faces
    fig, ax = plt.subplots()
    # colours so each cluster has it's own one
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    for i, img in enumerate(images):
        label = labels[i]
        ax.scatter(embeddings[i][0], embeddings[i][1], c=colors[label % len(colors)])
        ax.annotate(f'{image_filenames[i]}', (embeddings[i][0], embeddings[i][1]))
    plt.show()

    # Remove the Outliers
    clustered_images.pop(0)
    clustered_images.sort(key=len, reverse=True)
    # Print the clustered images
    for i, cluster in enumerate(clustered_images):
        print(f'Cluster {i}: {len(cluster)} images')
        print(f'Cluster {i}: {cluster}')
        # for filename in cluster:
        #     img = cv2.imread(os.path.join(folder_path, filename))
        #     cv2.imshow(f'Cluster {i}', img)
        #     cv2.waitKey(0)
        # cv2.destroyAllWindows()


    clusters_dir = 'Clusters'
    empty_dir(clusters_dir)

    # Create a subdirectory for each cluster and copy the corresponding images
    for i, cluster in enumerate(clustered_images):
        cluster_dir = os.path.join(clusters_dir, f'Cluster{i}')
        os.mkdir(cluster_dir)
        for filename in cluster:
            img_path = os.path.join(cluster_dir, filename)
            shutil.copyfile(os.path.join(folder_path, filename), img_path)
