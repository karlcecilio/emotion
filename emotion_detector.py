# USAGE
# python emotion_detector.py --cascade haarcascade_frontalface_default.xml --model ./datasets/fer2013/checkpoints/epoch_15.hdf5 --video ./datasets/video/test.mp4

from tensorflow.keras.utils import img_to_array
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
model = load_model(args["model"],safe_mode=False)

EMOTIONS = ["angry", "fear", "happy", "sad", "surprise", "neutral"]

camera = cv2.VideoCapture(args["video"])

while True:
    (grabbed, frame) = camera.read()
    if not grabbed:
        break

    # ---------- 人脸检测部分（采用优化后的参数 + 几何过滤，保持检测稳定） ----------
    frame = imutils.resize(frame, width=300)  # 检测用300宽，稳定性好
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 用于显示的画板（高分辨率，保证文字清晰）
    canvas = np.zeros((420, 600, 3), dtype="uint8")  # 宽度600，每个情绪条60高
    frameClone = frame.copy()

    # 优化后的 detectMultiScale 参数
    rects = detector.detectMultiScale(gray,
                                      scaleFactor=1.05,  # 缩小步长，检测更精细
                                      minNeighbors=8,  # 提高邻居数，过滤虚假人脸
                                      minSize=(60, 60),  # 提高最小尺寸，避免小噪点
                                      flags=cv2.CASCADE_SCALE_IMAGE)

    if len(rects) > 0:
        # 几何过滤：宽高比（正面人脸应在0.8~1.2之间）
        rects = [r for r in rects if 0.8 <= r[2] / r[3] <= 1.2]
        # 几何过滤：面积过滤（不能太小或太大，相对于画面）
        area_frame = gray.shape[0] * gray.shape[1]
        rects = [r for r in rects if 0.005 < (r[2] * r[3]) / area_frame < 0.4]

        if len(rects) > 0:
            # 取面积最大的人脸（原始逻辑）
            rect = sorted(rects, key=lambda x: x[2] * x[3], reverse=True)[0]
            (fx, fy, fw, fh) = rect

            roi = gray[fy:fy + fh, fx:fx + fw]
            roi = cv2.resize(roi, (48, 48))
            roi = roi.astype("float") / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            preds = model.predict(roi)[0]
            print("[INFO] ----",preds.argmax(), preds)
            label = EMOTIONS[preds.argmax()]

            # ---------- 绘制概率条（高清晰）----------
            for i, (emotion, prob) in enumerate(zip(EMOTIONS, preds)):
                text = "{}: {:.2f}%".format(emotion, prob * 100)
                w = int(prob * 600)
                y_top = i * 60 + 5
                y_bottom = y_top + 55
                cv2.rectangle(canvas, (5, y_top), (w, y_bottom), (0, 0, 255), -1)
                cv2.putText(canvas, text, (10, y_top + 38), cv2.FONT_HERSHEY_SIMPLEX,
                            1.0, (255, 255, 255), 2)

            # 在原始帧（300宽）上画人脸框和标签
            cv2.putText(frameClone, label, (fx, fy - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 0, 255), 2)
            cv2.rectangle(frameClone, (fx, fy), (fx + fw, fy + fh), (0, 0, 255), 2)

    # ---------- 显示部分：将视频帧放大到600宽，与画板拼接 ----------
    h, w = frameClone.shape[:2]
    frame_large = cv2.resize(frameClone, (600, int(600 * h / w)))
    combined = np.vstack([frame_large, canvas])
    cv2.imshow("Emotion Detection", combined)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()