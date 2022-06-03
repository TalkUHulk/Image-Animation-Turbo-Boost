import requests
import numpy as np
import cv2
import base64
import ujson


def func(cv_img, is_source, is_initial, api="http://0.0.0.0:12581/fom"):
    image = cv2.imencode('.png', cv_img)[1]
    image_base64 = base64.b64encode(image).decode('utf8')
    body = ujson.dumps(
        {"image_raw": image_base64,
         "is_source": is_source,
         "is_initial": is_initial})

    response = requests.post(url=api, data=body, headers={
        "content-type": "application/json"
    }, timeout=10)
    if not response.ok:
        print("error", response.text)
        return None
    return response.json()


# source_path = "Monalisa.png"
source_path = "images/jay.png"
video_path = "video.mp4"

source_image = cv2.imread(source_path)
source_image = cv2.resize(source_image, (256, 256))


cap = cv2.VideoCapture(video_path)

result = func(source_image, is_source=True, is_initial=False)

count = 0
while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    count += 1
    if ret:
        frame_face = frame[100:650, 450: 1000, :]
        frame_face = cv2.resize(frame_face, (256, 256))

        if count < 25:
            continue
        if count == 25:
            result = func(frame_face, is_source=False, is_initial=True)
        else:
            result = func(frame_face, is_source=False, is_initial=False)
        image_raw = result["image_raw"]
        print("code:", result["code"])
        image = base64.b64decode(image_raw)
        image = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR)

        cv2.imshow("demo", image[..., ::-1])
        cv2.imshow("source", frame)
        cv2.waitKey(1)

    else:
        break

cap.release()