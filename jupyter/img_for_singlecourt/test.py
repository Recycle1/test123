import os
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# 获取当前目录下的所有jpeg图片
image_paths = [f for f in os.listdir('.') if f.endswith('.jpeg')]

# 定义角点
p1, p2, p3, p4 = (183, 337), (1099, 337), (183, 719), (1099, 719)

# 用于存储所有图像中检测到的线条
all_lines = []

# 处理每张图片
for path in image_paths:
    img_org = cv.imread(path)
    img_gray = cv.cvtColor(img_org, cv.COLOR_BGR2GRAY)
    ret, th1 = cv.threshold(img_gray, 150, 255, cv.THRESH_BINARY)
    roi = th1[p1[1]:p4[1], p1[0]:p4[0]]
    edges_roi = cv.Canny(roi, 50, 150, apertureSize=3)

    # 执行霍夫线变换
    linesP = cv.HoughLinesP(edges_roi, rho=1, theta=np.pi/180, threshold=120, minLineLength=10, maxLineGap=14)

    # 存储检测到的线条
    if linesP is not None:
        for line in linesP:
            all_lines.append(line[0])

# 创建一个新图像
new_img = np.zeros((p4[1]-p1[1], p4[0]-p1[0], 3), np.uint8)

# 在新图像上绘制所有检测到的线条
for line in all_lines:
    x1, y1, x2, y2 = line
    cv.line(new_img, (x1, y1), (x2, y2), (0, 255, 0), 3)

# 显示和保存结果
plt.figure(figsize=(10, 8))
plt.imshow(cv.cvtColor(new_img, cv.COLOR_BGR2RGB))
plt.title('Detected Lines in All Images'), plt.xticks([]), plt.yticks([])
plt.show()