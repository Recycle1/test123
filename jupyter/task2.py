import cv2 as cv
import numpy as np

def on_trackbar_change(_):
    # 当滑动条改变时，更新HSV阈值，然后进行霍夫线变换并显示结果
    h_min = cv.getTrackbarPos('H low', 'image')
    h_max = cv.getTrackbarPos('H high', 'image')
    s_min = cv.getTrackbarPos('S low', 'image')
    s_max = cv.getTrackbarPos('S high', 'image')
    v_min = cv.getTrackbarPos('V low', 'image')
    v_max = cv.getTrackbarPos('V high', 'image')
    hsv_low = np.array([h_min, s_min, v_min])
    hsv_high = np.array([h_max, s_max, v_max])

    mask = cv.inRange(hsv_img, hsv_low, hsv_high)  # 创建HSV掩码
    roi = mask[p1[1]:p4[1], p1[0]:p4[0]]
    edges_roi = cv.Canny(roi, 50, 150, apertureSize=3)

    # 执行霍夫线变换
    linesP = cv.HoughLinesP(edges_roi, rho=1, theta=np.pi/90, threshold=105, minLineLength=5, maxLineGap=50)
    result_img = np.zeros_like(hsv_img)  # 创建一个用于绘制线条的空图像

    # 在新图像上绘制所有检测到的线条
    if linesP is not None:
        for line in linesP:
            x1, y1, x2, y2 = line[0]
            cv.line(result_img, (x1 + p1[0], y1 + p1[1]), (x2 + p1[0], y2 + p1[1]), (0, 255, 0), 3)
    # 显示结果
    cv.imshow('Detected Lines', result_img)


# 读取图片并转换为HSV
image = cv.imread('3.jpeg')
hsv_img = cv.cvtColor(image, cv.COLOR_BGR2HSV)

# 定义角点
p1, p2, p3, p4 = (235, 325), (1042, 325), (235, 684), (1042, 684)

# 创建滑动条窗口
cv.namedWindow('image')
cv.createTrackbar('H low', 'image', 35, 255, on_trackbar_change)
cv.createTrackbar('H high', 'image', 90, 255, on_trackbar_change)
cv.createTrackbar('S low', 'image', 43, 255, on_trackbar_change)
cv.createTrackbar('S high', 'image', 255, 255, on_trackbar_change)
cv.createTrackbar('V low', 'image', 35, 255, on_trackbar_change)
cv.createTrackbar('V high', 'image', 255, 255, on_trackbar_change)

cv.imshow('BGR', image)  # 显示原图
on_trackbar_change(None)  # 初始更新

while True:
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cv.destroyAllWindows()