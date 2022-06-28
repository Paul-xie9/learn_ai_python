import cv2

image_path = '/learn_cv_tf/resource/image/videoImage/'
# 视频分解图片
# 1 加载；2 获取视频信息；3 解码取到单帧视频； 4 展示或者写入
multiplePeoples_Path = '/filehome/PythonProjects/learn_ai_python/video/multiple_peoples.mp4'
# 获取一个视频打开cap
cap = cv2.VideoCapture(multiplePeoples_Path)
# 判断是否打开
isOpened = cap.isOpened()
print('视频是否打开：', isOpened)
# 获取视频的帧率
fps = cap.get(cv2.CAP_PROP_FPS)
# 获取视频的宽度和高度
width = int(cv2.CAP_PROP_FRAME_WIDTH)
height = int(cv2.CAP_PROP_FRAME_HEIGHT)
print('帧率：', fps)
print('宽度：', width, '；高度：', height)
i = 0
while isOpened:
    i += 1
    # 读取每一张的是否成功和内容
    (flag, frame) = cap.read()
    fileName = 'image' + str(i) + '.jpg'
    print('fileName：', fileName, 'flag:', flag)
    if flag:
        cv2.imwrite(image_path + fileName, frame, [cv2.IMWRITE_JPEG_QUALITY, 100])
    else:
        break
print('end')
