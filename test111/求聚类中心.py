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


# 计算聚类簇的中心点
centers = []
for label in set(labels):
    if label != -1:
        cluster_points = X[labels == label]
        center = np.mean(cluster_points, axis=0)
        centers.append(center)
plt.imshow(image)
colors = ['red', 'green', 'blue', 'yellow', 'orange', 'purple', 'cyan', 'magenta']
# 打印聚类簇的中心点
for i, center in enumerate(centers):
    print(f"Cluster {i+1} center: {center}")
    x, y = center
    plt.plot(x, y, marker='o', markersize=2, color=colors[i % len(colors)])



# 设置图形标题和坐标轴
plt.title('DBSCAN Clustering on Image')
plt.axis('off')

# 显示图形
plt.show()