import cv2

path = '../resource/Haar/haarcascade_eye.xml'

# 1.加载xml信息;xml里面是训练好了的眼睛和脸部的数据
face_xml = cv2.CascadeClassifier('../resource/Haar/haarcascade_frontalface_default.xml')
eye_xml = cv2.CascadeClassifier('../resource/Haar/haarcascade_eye.xml')
# 2.加载图片
img = cv2.imread('../resource/image/lenna.png')
cv2.imshow('src', img)
# 3.计算haar特征并灰度化， haar特征已经被cv计算了，不需要再次计算
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 4.检测脸部
faces = face_xml.detectMultiScale(gray, 1.3, 5)
print('face:', len(faces))
# 画出脸部区域
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    roi_face_gray = gray[y:y + h, x:x + w]
    roi_face_color = img[y:y + h, x:x + w]
    eyes = eye_xml.detectMultiScale(roi_face_gray)
    print('eyes', len(eyes))
    for (e_x, e_y, e_w, e_h) in eyes:
        cv2.rectangle(roi_face_color, (e_x, e_y), (e_x + e_w, e_y + e_h), (0, 0, 255), 2)
cv2.imshow('detect face(haar+adaboost)', img)
cv2.waitKey(0)
