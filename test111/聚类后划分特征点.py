import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from PIL import Image
import cv2
# 加载图像
image = cv2.imread("E:\\code_pipei\\new_pipei\\1.jpg")

import csv
with open('E:\\code_pipei\\new_pipei\\pipe_keypoint\\kp0.csv','r') as file:
    csv_reader = csv.reader(file)
    next(csv_reader)
    coordinates = []
    for row in csv_reader:
        x = float(row[0])
        y = float(row[1])
        coordinates.append((x, y))
X = np.array(coordinates)

# 创建DBSCAN对象并进行聚类
dbscan = DBSCAN(eps=30, min_samples=2)
labels = dbscan.fit_predict(X)

cluster_coordinates = {}
for i, label in enumerate(labels):
    if label != -1:
        if label not in cluster_coordinates:
            cluster_coordinates[label] = []
        cluster_coordinates[label].append(X[i])
for label, coordinates in cluster_coordinates.items():
    print(f"Cluster {label} coordinates: {coordinates}")
