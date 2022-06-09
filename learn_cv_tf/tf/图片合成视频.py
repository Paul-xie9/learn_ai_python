import cv2

ImagePath = '/filehome/PythonProjects/learn_ai_python/image/videoImage/'

img = cv2.imread(ImagePath + 'image1.jpg')
imageInfo = img.shape
size = imageInfo[:2]
print('size:', size)
# 写入对象
videoWriter = cv2.VideoWriter('/filehome/PythonProjects/learn_ai_python/video'+'2.mp4v', -1, 5, size)
for i in range(1, 1281):
    fileName = 'image' + str(i) + '.jpg'
    img = cv2.imread(fileName)
    # 写入图片
    videoWriter.write(img)
print('end!')
