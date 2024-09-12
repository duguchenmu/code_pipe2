
import cv2

def match_ss(des0, des1):
    # 创建BFMatcher对象
    bf = cv2.BFMatcher()

    # 使用KNN匹配特征
    matches = bf.knnMatch(des0, des1, k=2)
    print(matches)


