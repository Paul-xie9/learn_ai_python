import cv2
import numpy as np

# 1.准备数据
rand1 = np.array([[155, 43], [156, 50], [164, 53], [168, 56], [172, 60], [170, 60]])
rand2 = np.array([[152, 53], [156, 55], [160, 56], [168, 56], [172, 64], [176, 65]])
# 2. 标签; 0女生 1男生
label = np.array([[0], [0], [0], [0], [0], [0], [1], [1], [1], [1], [1], [1]])
# 3.数据转换; 监督学习； 0负样本 1正样本
data = np.vstack((rand1, rand2))
data = np.array(data, dtype='float32')
# 4.训练
svm = cv2.ml.SVM_create()  # 创建支持向量机
# 设置属性
svm.setType(cv2.ml.SVM_C_SVC)
svm.setKernel(cv2.ml.SVM_LINEAR)  # 线性分类器
svm.setC(0.01)
# 训练
result = svm.train(data, cv2.ml.ROW_SAMPLE, label)
# 预测
pt_data = np.vstack([[167, 55], [162, 57]])  # 0 1
pt_data = np.array(pt_data, dtype='float32')
print('pt_data:\n', pt_data)
(par1, par2) = svm.predict(pt_data)
print('par:\n', par1, par2)
