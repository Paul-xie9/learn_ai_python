import numpy as np
import cv2

Path = '/learn_cv_tf/resource/image/lenna.png'
newImageInfo = (500, 500, 3)
dst = np.zeros(newImageInfo, np.uint8)
# ==========================绘制线段===============================
'''
绘制的目标图片，开始位置，结束位置，线段颜色 线条宽度 线条类型
'''
cv2.line(dst, (100, 100), (400, 400), (0, 0, 255))
cv2.line(dst, (100, 200), (400, 200), (0, 255, 255), 15)
cv2.line(dst, (100, 300), (400, 300), (0, 255, 0), 15, cv2.LINE_AA)
# 绘制三角形
cv2.line(dst, (20, 200), (350, 350), (0, 255, 0), 5, cv2.LINE_AA)
cv2.line(dst, (350, 350), (150, 150), (0, 255, 0), 5, cv2.LINE_AA)
cv2.line(dst, (150, 150), (20, 200), (0, 255, 0), 5, cv2.LINE_AA)
cv2.imshow('line', dst)

# ==========================绘制多边形===============================
dst = np.zeros(newImageInfo, np.uint8)
'''
绘制的目标图片 左上角坐标 右下角坐标 填充颜色 是否填充（-1表示填充，其他表示不填充）
'''
cv2.rectangle(dst, (50, 100), (200, 300), (255, 0, 0), -1)
cv2.imshow('rectangle', dst)
'''
绘制的目标图片 圆心 半径 颜色 是否填充
'''
cv2.circle(dst, (250, 250), 50, (0, 255, 0), 1)
cv2.imshow('circle', dst)
'''
绘制的目标图片 椭圆的圆心 轴 偏转角度 起始角度 终止角度 颜色 是否填充
'''
cv2.ellipse(dst, (256, 256), (150, 100), 0, 0, 180, (255, 255, 0), -1)
cv2.imshow('ellipse', dst)
'''
任意多边形
'''
points = np.array([[150, 150], [140, 140], [200, 170], [250, 250], [150, 150]], np.int32)
points = points.reshape(-1, 1, 2)
cv2.polylines(dst, [points], True, (0, 255, 255))
cv2.imshow('polylines', dst)

# ==========================文字图片绘制===============================
image = cv2.imread(Path, 1)
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.rectangle(image, (200, 100), (500, 400), (0, 255, 0), 3)
cv2.putText(image, 'this is text', (100, 300), font, 1, (200, 100, 255), 2, cv2.LINE_AA)
cv2.imshow('text', image)

image = cv2.imread(Path, 1)
height = int(image.shape[0] * 0.2)
width = int(image.shape[1] * 0.2)
imageResize = cv2.resize(image, (width, height))
for i in range(0, height):
    for j in range(0, width):
        image[i+200, j+350] = imageResize[i, j]
cv2.imshow('picture', image)

cv2.waitKey(0)
