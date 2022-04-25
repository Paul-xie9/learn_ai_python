import cv2
import numpy as np
img = cv2.imread('/filehome/PythonProjects/learn_ai_python/image/lenna.png', 1)
imgInfo = img.shape
height = imgInfo[0]
width = imgInfo[1]
mode = imgInfo[2]
# 放大/缩小； 等比例缩放：非等比例缩放# 缩小一半
dstHeight = int(height * 0.5)
dstWidth = int(width * 0.5)
img = cv2.resize(img, (dstWidth, dstHeight))
imgInfo = img.shape
height = imgInfo[0]
width = imgInfo[1]
mode = imgInfo[2]
cv2.imshow('img', img)
# 最临近域插值，双线性插值，像素关系重采样，立方插值
dst = cv2.resize(img, (dstWidth, dstHeight))
cv2.imshow('image', dst)
'''双线性插值算法原理实现'''
dstImage = np.zeros((dstHeight, dstWidth, 3), np.uint8)
for i in range(0, dstHeight):
    for j in range(0, dstWidth):
        iNew = int(i*(height*1.0/dstHeight))
        jNew = int(j*(width*1.0/dstWidth))
        dstImage[i, j] = img[iNew, jNew]
cv2.imshow('dst', dstImage)
'''图片剪切像素x: 100 - 200像素y: 200 - 300'''
dstImage = img[100:200, 200:300]
cv2.imshow('dst', dstImage)
'''位移矩阵'''
matShift = np.float_([[1, 0, 100], [0, 1, 200]])
dst = cv2.warpAffine(img, matShift, (height, width))
cv2.imshow('dst', dst)
'''源码位移图片'''
dst = np.zeros(img.shape, np.uint8)
for i in range(0, height):
    for j in range(0, width - 100):
        dst[i, j + 100] = img[i, j]
cv2.imshow('dst', dst)
'''利用矩阵缩放图片'''
matScale = np.float32([[0.5, 0, 0], [0, 0.5, 0]])
dst = cv2.warpAffine(img, matScale, (int(width / 2), int(height / 2)))
cv2.imshow('dst', dst)
'''图片镜像'''
deep = imgInfo[2]
newImgInfo = (height * 2, width, deep)
dst = np.zeros(newImgInfo, np.uint8)
for i in range(height):
    for j in range(width):
        dst[i, j] = img[i, j]
        # 镜像部分
        dst[height * 2 - i - 1, j] = img[i, j]
        dst[height, i] = (0, 0, 255)
cv2.imshow('dst', dst)
'''仿射变换'''
matSrc = np.float32([[0, 0], [0, height - 1], [width - 1, 0]])
matDst = np.float32([[20, 20], [300, height - 100], [width - 100, 100]])
matAffine = cv2.getAffineTransform(matSrc, matDst)
dst = cv2.warpAffine(img, matAffine, (width, height))
cv2.imshow('dst', dst)
'''图片旋转'''
# 中心点，旋转角度，缩放系数
matRotate = cv2.getRotationMatrix2D((height * 0.5, width * 0.5), 45, 0.5)
dst = cv2.warpAffine(img, matRotate, (height, width))
cv2.imshow('dst', dst)
cv2.waitKey(0)
print('end!')
