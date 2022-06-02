import matplotlib.pyplot as plt

from image import myInImageUtils
import numpy as np
import cv2

imageFile = myInImageUtils.image_in('lenna')


# ===========================直方图=====================================
def myImageHist(image, type):
    color = (255, 255, 255)
    windowName = 'Gray'
    if type == 31:
        # 蓝色直方图
        color = (255, 0, 0)
        windowName = 'B Hist'
    elif type == 32:
        # 绿色直方图
        color = (0, 255, 0)
        windowName = 'G Hist'
    elif type == 33:
        # 红色直方图
        color = (0, 0, 255)
        windowName = 'R Hist'
    # 图像数据 直方图通道 mask蒙版 直方图的size 直方图中各个像素值
    hist = cv2.calcHist([image], [0], None, [256], [0.0, 255.0])
    minV, maxV, minL, maxL = cv2.minMaxLoc(hist)
    histImg = np.zeros([256, 256, 3], np.uint8)
    for h in range(256):
        intenNormal = int(hist[h] * 256 / maxV)
        cv2.line(histImg, (h, 256), (h, 256 - intenNormal), color)
    cv2.imshow(windowName, histImg)
    return histImg


image = cv2.imread(imageFile, 1)
channels = cv2.split(image)
for i in range(0, 3):
    myImageHist(channels[i], 31 + i)

# ===========================直方图均衡化=====================================
image = cv2.imread(imageFile)
# 灰度图片的直方图均衡化
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 将彩色图片转换为灰度图
cv2.imshow('grayImg', gray)
result = cv2.equalizeHist(gray)
cv2.imshow('grayHist', result)

# 彩色图片的直方图均衡化
colorImg = cv2.imread(imageFile)
cv2.imshow('colorImg', colorImg)
(b, g, r) = cv2.split(colorImg)  # 分解通道
# 对各个通道进行直方图均衡化
bH = cv2.equalizeHist(b)
gH = cv2.equalizeHist(g)
rH = cv2.equalizeHist(r)
# 将均衡化之后的数据合并在一起
result = cv2.merge((bH, gH, rH))
cv2.imshow('colorHist', result)

# YUV图片的直方图均衡化
colorImg = cv2.imread(imageFile)
# 将彩色图片转换为yuv图
YUVImg = cv2.cvtColor(colorImg, cv2.COLOR_BGR2YUV)
channelYUV = cv2.split(YUVImg)
channelYUV[0] = cv2.equalizeHist(channelYUV[0])
channels = cv2.merge(channelYUV)
result = cv2.cvtColor(channels, cv2.COLOR_YUV2BGR)
cv2.imshow('YUVHist', result)

# ==========================图片修补============================================
# 将图片破坏 画线破坏
img = cv2.imread(imageFile, 1)
for i in range(200, 300):
    img[i, 200] = [255, 0, 0]
    img[i, 200 - 1] = [255, 0, 0]
    img[i, 200 + 1] = [255, 0, 0]
for j in range(150, 300):
    img[250, j] = [0, 2555, 0]
    img[250 + 1, j] = [0, 255, 0]
    img[250 - 1, j] = [0, 2555, 0]
cv2.imwrite(myInImageUtils.fileHead + 'damaged.png', img)
# 读取到被破坏的图片
damagedImg = cv2.imread(myInImageUtils.image_in('damaged'), 1)
cv2.imshow('damaged', damagedImg)
imgInfo = damagedImg.shape
height = imgInfo[0]
width = imgInfo[1]
paint = np.zeros((height, width, 1), np.uint8)
for i in range(200, 300):
    paint[i, 200] = 255
    paint[i, 200 - 1] = 255
    paint[i, 200 + 1] = 255
for j in range(150, 300):
    paint[250, j] = 255
    paint[250 + 1, j] = 255
    paint[250 - 1, j] = 255
cv2.imshow('paint', paint)
repair = cv2.inpaint(damagedImg, paint, 3, cv2.INPAINT_TELEA)
cv2.imwrite(myInImageUtils.fileHead + 'repair.png', repair)
cv2.imshow('repair', repair)

# ==========================灰度直方图源码============================================
# 统计每个像素灰度出现的概率 0-255
img = cv2.imread(imageFile, 1)
imgInfo = img.shape
(height, width) = imgInfo[:2]
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
count = np.zeros(256, np.float)
for i in range(0, height):
    for j in range(0, width):
        pixel = gray[i, j]
        index = int(pixel)
        count[index] = count[index] + 1
for i in range(0, 255):
    count[i] = count[i] / (height * width)
x = np.linspace(0, 255, 256)
y = count
plt.bar(x, y, 0.9, alpha=1, color='b')
plt.show()

# ==========================灰度直方图源码============================================
# 计算每个像素的概率
sum1 = float(0)
for i in range(0, 255):
    sum1 = sum1 + count[i]
    count[i] = sum1
# print(count)
# 计算映射表
map1 = np.zeros(256, np.uint16)
for i in range(0, 255):
    map1[i] = np.uint16(count[i] * 255)
# 映射
for i in range(0, height):
    for j in range(0, width):
        pixel = gray[i, j]
        gray[i, j] = map1[pixel]
cv2.imshow('dst', gray)

# ==========================彩色直方图源码============================================
# 统计每个色彩像素出现的概率
img = cv2.imread(imageFile, 1)
imgInfo = img.shape
(height, width) = imgInfo[:2]
count_b = np.zeros(256, np.float)
count_g = np.zeros(256, np.float)
count_r = np.zeros(256, np.float)
for i in range(0, height):
    for j in range(0, width):
        (b, g, r) = img[i, j]
        index_b = int(b)
        index_g = int(g)
        index_r = int(r)
        count_b[index_b] += 1
        count_g[index_g] += 1
        count_r[index_r] += 1
for i in range(0, 255):
    count_b[i] = count_b[i] / (height * width)
    count_g[i] = count_g[i] / (height * width)
    count_r[i] = count_r[i] / (height * width)
x = np.linspace(0, 255, 256)
y1 = count_b
plt.figure()
plt.bar(x, y1, 0.9, alpha=1, color='b')
y2 = count_g
plt.figure()
plt.bar(x, y2, 0.9, alpha=1, color='g')
y3 = count_r
plt.figure()
plt.bar(x, y3, 0.9, alpha=1, color='r')
plt.show()

# ==================================亮度增强====================================
# p = p + 增加的亮度
img = cv2.imread(imageFile, 1)
imgInfo = img.shape
(height, width) = imgInfo[:2]
dst = np.zeros(imgInfo, np.uint8)
for i in range(0, height):
    for j in range(0, width):
        (b, g, r) = img[i, j]
        bb = int(b) + 40
        gg = int(g) + 40
        rr = int(r) + 40
        if bb > 255:
            bb = 255
        if gg > 255:
            gg = 255
        if rr > 255:
            rr = 255
        dst[i, j] = (bb, gg, rr)
cv2.imshow('brightness1', dst)
# p = p*1.2 + 增加的亮度
img = cv2.imread(imageFile, 1)
imgInfo = img.shape
(height, width) = img.shape[:2]
dst = np.zeros(imgInfo, np.uint8)
for i in range(0, height):
    for j in range(0, width):
        (b, g, r) = img[i, j]
        bb = int(b * 1.2) + 40
        gg = int(g * 1.2) + 40
        if bb > 255:
            bb = 255
        if gg > 255:
            gg = 255
        dst[i, j] = (bb, gg, r)
cv2.imshow('brightness2', dst)

# ==================================磨平美白====================================
# 双边滤波
img = cv2.imread(imageFile, 1)
dst = cv2.bilateralFilter(img, 15, 35, 35)
cv2.imshow('rubdown', dst)

# ==================================高斯滤波====================================
img = cv2.imread(imageFile, 1)
dst = cv2.GaussianBlur(img, (5, 5), 1.5)
cv2.imshow('GaussianBlur', dst)

# ==================================均值滤波====================================
img = cv2.imread(imageFile, 1)
imgInfo = img.shape
(height, width) = img.shape[:2]
dst = np.zeros(imgInfo, np.uint8)
for i in range(3, height - 3):
    for j in range(3, width - 3):
        sum_b = int(0)
        sum_g = int(0)
        sum_r = int(0)
        for m in range(-3, 3):
            for n in range(-3, 3):
                (b, g, r) = img[i + m, j + n]
                sum_b += int(b)
                sum_g += int(g)
                sum_r += int(r)
        b = np.uint8(sum_b / 36)
        g = np.uint8(sum_g / 36)
        r = np.uint8(sum_r / 36)
        dst[i, j] = (b, g, r)
cv2.imshow('avg GaussianBlur', dst)

# ==================================中值滤波====================================
img = cv2.imread(imageFile, 1)
imgInfo = img.shape
(height, width) = img.shape[:2]
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
dst = np.zeros(imgInfo, np.uint8)
collect = np.zeros(9, np.uint8)
for i in range(1, height - 1):
    for j in range(1, width - 1):
        k = 0
        for m in range(-1, 2):
            for n in range(-1, 2):
                gray = img[i + m, j + n]
                collect[k] = gray
                k += 1
        for k in range(0, 9):
            p1 = collect[k]
            for t in range(k + 1, 9):
                if p1 < collect[t]:
                    mid = collect[t]
                    collect[t] = p1
                    p1 = mid
        dst[i, j] = collect[4]
cv2.imshow('mid smoothing', dst)

cv2.waitKey(0)
