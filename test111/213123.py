import cv2
from matplotlib import pyplot as plt
import numpy as np
img = cv2.imread('E:\\code_pipei\\new_pipei\\1.jpg')
img2 = cv2.imread('E:\\code_pipei\\new_pipei\\2.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# # SURF或SIFT特征
# sift = cv2.xfeatures2d.SIFT_create()
# surf = cv2.xfeatures2d.SURF_create(400)
# surf.setExtended(True)
sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(gray, mask=None)
kp2, des2 = sift.detectAndCompute(gray2, mask=None)

anot1 = cv2.drawKeypoints(gray, kp1, None)
anot2 = cv2.drawKeypoints(gray2, kp2, None)
plt.subplot(121)
plt.imshow(anot1)
plt.subplot(122)
plt.imshow(anot2)

# # MATCH
matcher = cv2.BFMatcher()
raw_matches = matcher.knnMatch(des1, des2, k=2)
good_matches = []
for m1, m2 in raw_matches:
    #  如果最接近和次接近的比值大于一个既定的值，那么我们保留这个最接近的值，认为它和其匹配的点为good_match
    if m1.distance < 0.85 * m2.distance:
        good_matches.append([m1])
# good_matches = sorted(raw_matches, key=lambda x: x[0].distance)[:300]

# # RANSAC
assert len(good_matches) > 4, "Too few matches."
kp1_array = np.float32([kp1[m[0].queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
kp2_array = np.float32([kp2[m[0].trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
H, status = cv2.findHomography(kp1_array, kp2_array, cv2.RANSAC, ransacReprojThreshold=4)
good_matches = [good_matches[i] for i in range(len(good_matches)) if status[i] == 1]
imgOut = cv2.warpPerspective(gray2, H, (gray.shape[1], gray.shape[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
print(H)  # 变换矩阵H，透视变换？
# plt.figure()
# plt.imshow(imgOut)
# cv2.findFundamentalMat()  # 用于3D

matches = cv2.drawMatchesKnn(anot1, kp1, anot2, kp2, good_matches, None, flags = 2)

plt.figure()
plt.imshow(matches)

plt.show()
