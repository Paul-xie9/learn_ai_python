import cv2
import numpy as np

# 1.参数设置
PosNum = 820  # 正样本
NegNum = 1931  # 负样本
winSize = (64, 128)  # win大小
blockSize = (16 * 16)
blockStride = (8 * 8)
cellSize = (8 * 8)
nBin = 9
# 2.hog创建
hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nBin)
# 3.svm创建
svm = cv2.ml.SVM_create()
# 4.计算hog
featureNum = int(((128 - 16) / 8 + 1) * ((64 - 16) / 8 + 1) * 4 * 9)
print('featureNum:', featureNum)
featureArray = np.zeros(((PosNum + NegNum), featureNum), np.float32)
labelArray = np.zeros(((PosNum + NegNum), 1), np.int32)
# 5.标记
# 处理正样本
for i in range(0, PosNum):
    fileName = '../resource/image/pos/' + str(i + 1) + '.jpg'
    img = cv2.imread(fileName)
    hist = hog.compute(img, (8 * 8))
    print('hist(+):', hist)
    for j in range(0, featureNum):
        featureArray[i, j] = hist[j]
    labelArray[i, 0] = 1
# 处理负样本
for i in range(0, PosNum):
    fileName = '../resource/image/neg' + str(i + 1) + '.jpg'
    img = cv2.imread(fileName)
    hist = hog.compute(img, (8 * 8))
    print('hist(-):', hist)
    for j in range(0, featureNum):
        featureArray[i + PosNum, j] = hist[j]
    labelArray[i + PosNum, 0] = -1
svm.setType(cv2.ml.SVM_C_SVC)
svm.setKernel(cv2.ml.SVM_LINEAR)
svm.setC(0.01)
# 6.训练
result = svm.train(featureArray, cv2.ml.ROW_SAMPLE, labelArray)
# 7.检测
alpha = np.zeros(1, np.float32)
print('alpha:', alpha)
rho = svm.getDecisionFunction(0, alpha)
print('rho:', rho)
alphaArray = np.zeros((1, 1), np.float32)
supportVArray = np.zeros((1, featureNum), np.float32)  # 支持向量基数
resultArray = np.zeros((1, featureNum), np.float32)
alphaArray[0, 0] = alpha
resultArray = -1 * alphaArray * supportVArray
# detect
myDetect = np.zeros((3781, np.float32))
for i in range(0, 3780):
    myDetect[i] = resultArray[0, i]
myDetect[3780] = rho[0]
# 构建hog
myHog = cv2.HOGDescriptor()
myHog.setSVMDetector(myDetect)
imageSrc = cv2.imread('../resource/image/lenna.png', 1)
objs = myHog.detectMultiScale(imageSrc, 0, (8, 8), (32, 32), 1.05, 2)
x = int(objs[0][0][0])
y = int(objs[0][0][1])
w = int(objs[0][0][2])
h = int(objs[0][0][3])
cv2.rectangle(imageSrc, (x, y), (x + w, y + w), (255, 0, 0), 2)
cv2.imshow('dst', imageSrc)
cv2.waitKey(0)
