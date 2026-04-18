# USAGE
# python emotion_detector.py --cascade haarcascade_frontalface_default.xml --model ./datasets/fer2013/checkpoints/epoch_15.hdf5 --video ./datasets/video/test.mp4

from tensorflow.keras.utils import img_to_array          # 修正导入路径
from tensorflow.keras.models import load_model
import numpy as np
import argparse
import imutils
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-c", "--cascade", required=True, help="path to face cascade")
ap.add_argument("-m", "--model", required=True, help="path to emotion model")
ap.add_argument("-v", "--video", required=True, help="path to video file")
args = vars(ap.parse_args())

detector = cv2.CascadeClassifier(args["cascade"])
print("[INFO] loading model...")
model = load_model(args["model"])

# 修正情绪类别（FER2013 标准7类，根据你的模型输出调整）
EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

camera = cv2.VideoCapture(args["video"])

while True:
    (grabbed, frame) = camera.read()
    if not grabbed:
        break

    frame = imutils.resize(frame, width=300)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    canvas = np.zeros((220, 300, 3), dtype="uint8")   # 概率条画板
    frameClone = frame.copy()

    rects = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5,
                                      minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

    if len(rects) > 0:
        # 取面积最大的人脸
        rect = sorted(rects, reverse=True, key=lambda x: (x[2]-x[0])*(x[3]-x[1]))[0]
        (fx, fy, fw, fh) = rect

        roi = gray[fy:fy+fh, fx:fx+fw]
        roi = cv2.resize(roi, (48, 48))
        roi = roi.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)

        preds = model.predict(roi)[0]
        label = EMOTIONS[preds.argmax()]

        # 绘制概率条
        for i, (emotion, prob) in enumerate(zip(EMOTIONS, preds)):
            text = "{}: {:.2f}%".format(emotion, prob * 100)   # 修正：显示当前情绪名
            w = int(prob * 300)
            cv2.rectangle(canvas, (5, i*35+5), (w, i*35+35), (0, 0, 255), -1)
            cv2.putText(canvas, text, (10, i*35+23), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 2)

        # 在视频帧上画人脸框和标签
        cv2.putText(frameClone, label, (fx, fy-10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,255), 2)
        cv2.rectangle(frameClone, (fx, fy), (fx+fw, fy+fh), (0, 0, 255), 2)

    # 将视频帧和概率条垂直拼接
    combined = np.vstack([frameClone, canvas])
    cv2.imshow("Emotion Detection", combined)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()