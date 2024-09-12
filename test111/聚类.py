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

##
clusters = dbscan.labels_
n_clusters = len(set(clusters))
print(clusters)
print(n_clusters)
##


# 为每个聚类簇分配一个唯一的颜色
colors = ['red', 'green', 'blue', 'yellow', 'orange', 'purple', 'cyan', 'magenta']

# 绘制图像和聚类结果
plt.imshow(image)

# 将聚类结果叠加在图像上
for coord, label in zip(X, labels):
    x, y = coord
    if label == -1:
        # 绘制噪声点
        plt.plot(x, y, marker='o', markersize=2, color='black')
    else:
        # 绘制聚类簇
        plt.plot(x, y, marker='o', markersize=2, color=colors[label % len(colors)])

# 设置图形标题和坐标轴
plt.title('DBSCAN Clustering on Image')
plt.axis('off')

# 显示图形
plt.show()