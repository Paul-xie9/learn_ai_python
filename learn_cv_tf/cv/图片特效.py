import math
import random

import cv2
import numpy as np

filePath = '/learn_cv_tf/resource/image/lenna.png'
# ==================灰度化========================
'''
1.  cv2.imread读取的时候设置值
    1:彩色读取  2：灰度读取
'''
img_color = cv2.imread(filePath, 1)
cv2.imshow('img_color(cv2.imread(img, 1))', img_color)
img_gray = cv2.imread(filePath, 0)
cv2.imshow('img_gray(cv2.imread(img, 0))', img_gray)

'''
2.  cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
'''
res = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
cv2.imshow('img_gray(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))', res)

'''
3.  利用灰度图的特点：每个通道上的像素值相等
    即 R=G=B 将其转化为 （R+G+B）/3
'''
imgInfo = img_color.shape
height = imgInfo[0]
width = imgInfo[1]
# 创建一个和原图大小相等的零矩阵
res = np.zeros(imgInfo, np.uint8)
for i in range(height):
    for j in range(width):
        (b, g, r) = img_color[i, j]
        # 为了防止相加之后结果溢出，需要先将其转换为int类型
        gray = (int(b) + int(g) + int(r)) / 3
        res[i, j] = np.uint8(gray)
cv2.imshow('img_gray(R=G=B)', res)

'''
4.  心理学中的计算公式
    即 心理学中的灰度值 = b*0.114 + g*0.587 + r*0.299
'''
for i in range(height):
    for j in range(width):
        b, g, r = img_color[i, j]
        gray = b * 0.114 + g * 0.587 + r * 0.229
        res[i, j] = np.uint8(gray)
cv2.imshow('img_gray(b*0.114 + g*0.587 + r*0.299)', res)

# ==================颜色反转(地板效果)========================
'''
1.  颜色反转
'''
# 创建一个和原来灰度图片大小一样的零矩阵
res_gray = np.zeros(img_gray.shape, np.uint8)
# 创建一个和原来彩色图片大小一样的零矩阵
res_color = np.zeros(img_color.shape, np.uint8)
for i in range(height):
    for j in range(width):
        grayPixel = img_gray[i, j]
        res_gray[i, j] = 255 - grayPixel
        grayPixel = img_color[i, j]
        res_color[i, j] = 255 - grayPixel
cv2.imshow('img_convert(gray)', res_gray)
cv2.imshow('img_convert(color)', res_color)

# ==================马赛克========================
'''
1.  马赛克原理: 
        选中需要打马赛克的位像素区域，选中一个10*10的小方块(其中有6个像素方块)，
    用小方块的一个像素颜色(例如左上角的颜色)替换其他5个方块的颜色
'''
un_view = cv2.imread(filePath, 1)
for m in range(100, 300):
    for n in range(100, 200):
        if m % 10 == 0 and n % 10 == 0:
            for i in range(10):
                for j in range(10):
                    un_view[m + i, n + j] = un_view[m, n]
cv2.imshow('un_view', un_view)

# ==================毛玻璃========================
'''
1.  毛玻璃原理:
        和马赛克相比是产生随机的颜色
'''
random_plastic = cv2.imread(filePath, 1)
res = np.zeros(img_color.shape, np.uint8)
k = 4
for i in range(height - k):
    for j in range(width - k):
        # 产生的随即像素
        index = int(random.random() * k)
        res[i, j] = random_plastic[i + index, j + index]
cv2.imshow('random_plastic', res)

# ==================图片融合========================
'''
1.  原理:
        dst = image1*m + image2*(1-m)
'''
imagePath = '/filehome/PythonProjects/learn_ai_python/image/p1.png'
image1 = cv2.imread(filePath, 1)
image2 = cv2.imread(imagePath, 1)
img1Info = image1.shape
height1 = img1Info[0]
width1 = img1Info[0]
# ROI
roiH = int(height1)
roiW = int(width1)
image1ROI = image1[0:roiH, 0:roiW]
image2ROI = image2[0:roiH, 0:roiW]
res = np.zeros((roiH, roiW, 3), np.uint8)
res = cv2.addWeighted(image1ROI, 0.5, image2ROI, 0.5, 0)
cv2.imshow('blend', res)

# ==================边缘检测(canny)========================
'''
1.  边缘检测:
    需要灰度图，将其高斯滤波化，实现canny
'''
# 高斯滤波化
gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
imgG = cv2.GaussianBlur(gray, (3, 3), 0)
res = cv2.Canny(img_color, 50, 50)
cv2.imshow('margin check(canny) ', res)

# ==================边缘检测(算法原理)========================
# sobel算子模板 分为竖直方向与水平方向
'''
[            |    [
 1  2  1     |      1  0  -1
 0  0  0     |      2  0  -2
-1 -2 -1     |      1  0  -1
]            |              ]
'''
# 图片卷积: 当前像素乘以模板再求和
'''
[1 2 3 4] * [a b c b] = 1*a + 2*b + 3*c + 4*b = dst 
'''
# 阈值判决
'''
sqrt(a*a(竖直方向上卷积的结果) + b*b(水平方向上卷积的结果)) = f(幅值) > th(判决阈值) -> 表示为边缘
'''
gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
dst = np.zeros((height, width, 1), np.uint8)
for i in range(height - 2):
    for j in range(width - 2):
        # 竖直方向的卷积
        gy = gray[i, j] * 1 + gray[i, j + 1] * 2 + gray[i, j + 2] * 1 - gray[i + 2, j] * 1 - gray[i + 2, j + 1] * 2 - \
             gray[i + 2, j + 2] * 1
        # 水平方向的卷积
        gx = gray[i, j] + gray[i + 1, j] * 2 + gray[i + 2, j] - gray[i, j + 2] - gray[i + 1, j] * 2 - gray[i, j]
        # 计算梯度
        grad = math.sqrt(gy * gy + gx * gx)
        # 做阈值判决
        if grad > 20:
            dst[i, j] = 255
        else:
            dst[i, j] = 0
cv2.imshow('margin check(theory)', dst)

# ==================浮雕效果========================
'''
原理： newP = gray0 - gray1 + 150
'''
gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
dst = np.zeros((height, width, 1), np.uint8)
for i in range(height):
    for j in range(width - 1):
        grayP0 = int(gray[i, j])
        grayP1 = int(gray[i, j + 1])
        newP = grayP0 - grayP1 + 150
        if newP > 255:
            newP = 255
        if newP < 0:
            newP = 0
        dst[i, j] = newP
cv2.imshow('relief', dst)

# ==================颜色映射========================
# rgb -> new rgb ‘蓝色’（b=b*1.5；g=g*1.3）
image = cv2.imread(filePath, 1)
dst = np.zeros(imgInfo, np.uint8)
for i in range(width):
    for j in range(height):
        (b, g, r) = image[i, j]
        b = b * 1.5
        g = g * 1.3
        if b > 255:
            b = 255
        if g > 255:
            g = 255
        dst[i, j] = (b, g, r)
cv2.imshow('map', dst)

# ==================油画效果========================
gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
dst = np.zeros(imgInfo, np.uint8)
for i in range(4, height - 4):
    for j in range(4, width - 4):
        array1 = np.zeros(8, np.uint8)
        for m in range(-4, 4):
            for n in range(-4, 4):
                p1 = int(gray[i + m, j + n] / 32)
                array1[p1] = array1[p1] + 1
        currentMax = array1[0]
        l = 0
        for k in range(0, 8):
            if currentMax < array1[k]:
                currentMax = array1[k]
                l = k
        # 简化 均值
        for m in range(-4, 4):
            for n in range(-4, 4):
                if gray[i + m, j + n] >= (1 * 32) and gray[i + m, j + n] <= ((l + 1) * 32):
                    (b, g, r) = img_color[i + m, j + n]
        dst[i, j] = (b, g, r)
cv2.imshow('oil painting', dst)

cv2.waitKey(0)
print('program end!')
