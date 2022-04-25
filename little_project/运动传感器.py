import json
import requests
import cv2
import datetime


# 1.获取accessToken
# https请求方式: GET https://api.weixin.qq.com/cgi-bin/token?grant_type=client_credential&appid=APPID&secret=APPSECRET
def get_access_token(app_id, app_secret):
    url = f'https://api.weixin.qq.com/cgi-bin/token?grant_type=client_credential&appid=wx4bf0272bef36f55a&secret=1ad6c415a3a47ddf969b1315e459f335'
    resp = requests.get(url).json()
    access_token = resp.get('access_token')
    print(access_token)
    return access_token


# 2.发送客服消息
# http请求方式: POST https://api.weixin.qq.com/cgi-bin/message/custom/send?access_token=ACCESS_TOKEN
def send_wx_information(open_id, information):
    app_id = "wx4bf0272bef36f55a"
    app_secret = "1ad6c415a3a47ddf969b1315e459f335"
    access_token = get_access_token(app_id, app_secret)
    url = f'https://api.weixin.qq.com/cgi-bin/message/custom/send?access_token=' + access_token
    if information == '':
        information = '有人闯入你家'
    req_data = {
        "touser": open_id,
        "msgtype": "text",
        "text":
            {
                "content": information
            }
    }
    res = requests.post(url, data=json.dumps(req_data, ensure_ascii=False).encode('utf-8'))
    print(res)


def video_cv(open_id, information):
    camera = cv2.VideoCapture(0)
    background = None
    es = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 4))
    is_send_wx_msg = False
    while True:
        grabbed, frame = camera.read()
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_frame = cv2.GaussianBlur(gray_frame, (25, 25), 3)
        if background is None:
            background = gray_frame
            continue
        diff = cv2.absdiff(background, gray_frame)
        diff = cv2.threshold(diff, 50, 255, cv2.THRESH_BINARY)[1]
        diff = cv2.dilate(diff, es, iterations=3)

        contours, hierarchy = cv2.findContours(diff.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        is_detected = False
        for c in contours:
            if cv2.contourArea(c) < 2000:
                continue
            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            is_detected = True
            if not is_send_wx_msg:
                send_wx_information(open_id, information)
                is_send_wx_msg = True
        if is_detected:
            show_text = "undetected!"
            show_color = (0, 0, 255)
        else:
            show_text = "has detected thing!"
            show_color = (0, 255, 0)

        cv2.putText(frame, show_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, show_color, 2)
        cv2.putText(frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"),
                    (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, show_color, 1)
        cv2.imshow('video', frame)
        cv2.imshow('diff', diff)
        key = cv2.waitKey(1) & 0xFFf
        if key == ord('q'):
            break
    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    open_id = 'oCgSc5zwKnOREwO8EOsZRrMnVCa8'
    video_cv(open_id, '探测到物体了！')
