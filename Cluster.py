import dlib
import os
import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import face_recognition


# Load the images from a folder
folder_path = 'Film/Test'
images = []
for filename in os.listdir(folder_path):
    img = face_recognition.load_image_file(os.path.join(folder_path, filename))
    images.append(img)

# Detect and extract face embeddings from each image
embeddings = []
for img in images:
    face_location = (0, img.shape[0], img.shape[1], 0)
    face_encodings = face_recognition.face_encodings(img, [face_location])
    embeddings.append(face_encodings[0])

print(embeddings)

# Cluster the embeddings using k-means algorithm
kmeans = KMeans(n_clusters=3)
kmeans.fit(embeddings)
labels = kmeans.labels_

# Plot the clustered faces
fig, ax = plt.subplots()
for i, img in enumerate(images):
    label = labels[i]
    ax.scatter(embeddings[i][0], embeddings[i][1], c=label)
    ax.annotate(f'Face{i+1}', (embeddings[i][0], embeddings[i][1]))
plt.show()

