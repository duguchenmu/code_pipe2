import cv2
import numpy as np
from pytlsd import lsd
from sklearn.cluster import DBSCAN
def extract_screen(img):
    # 转换为灰度图像
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 计算灰度图像的亮度平均值
    mean_brightness = np.mean(img)

    # 根据亮度平均值计算阈值
    threshold = mean_brightness * 1.0

    # 对灰度图像进行阈值分割
    _, binary = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
    # 执行轮廓检测
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 创建一个空白图像，与原始图像大小和通道数相同
    result = np.zeros_like(binary)

    # 绘制轮廓到空白图像
    cv2.drawContours(result, contours, -1, (255), thickness=cv2.FILLED)

    masked_image = cv2.bitwise_and(img, img, mask=result)
    return masked_image

def edgepoint_elimination(img, kp):
    coordinates = []
    for k in kp:
        patch = cv2.getRectSubPix(img, (20, 20), k) #像素点20*20的周围
        if not np.any(patch == 0):
            coordinates.append(k)
    return np.array(coordinates)

def dbscan_cluster(kp):
    X = np.array(kp)
    dbscan = DBSCAN(eps=30, min_samples=2)
    labels = dbscan.fit_predict(X)
    cluster_coordinates = {}
    indices_dic = {}
    for i, label in enumerate(labels):
        if label != -1:
            if label not in cluster_coordinates:
                cluster_coordinates[label] = []
            cluster_coordinates[label].append(X[i])
    for label, coordinates in cluster_coordinates.items():
        # print(f"Cluster {label} coordinates_index: {np.where((kp == coordinates).all(axis=1))}")
        indices = []
        for coordinate in coordinates:
            indices.append(int(np.where((kp == coordinate).all(axis=1))[0]))
        indices_dic[label] = indices
    return indices_dic

def update_pred(pred, indices):
    pred_new = {}
    for k in indices:
        indice = indices[k]
        pred_new[f'keypoints{k}'] = pred['keypoints0'][indice]
        pred_new[f'keypoint_scores{k}'] = pred['keypoint_scores0'][indice]
        pred_new[f'descriptors{k}'] = pred['descriptors0'].T[indice].T
        # pred_new[f'all_descriptors{k}'] = pred['all_descriptors0'].T[indice].T
    return pred_new