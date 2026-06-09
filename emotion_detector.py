# USAGE
# python emotion_detector.py --cascade haarcascade_frontalface_default.xml --model ./datasets/fer2013/checkpoints/mini_xception_final_best_epoch_41.hdf5 --video ./datasets/video/test.mp4

from tensorflow.keras.utils import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import argparse
import imutils
import cv2
import tensorflow as tf

# ---------- 自定义损失函数（与训练时一致，用于加载模型）----------
def weighted_smooth_cce(y_true, y_pred):
    # 简化版：直接使用交叉熵（因为加载模型时不需要严格计算梯度）
    # 但为了与训练时一致，保留同样的函数体（若想精确复现，需复制训练代码中的完整定义）
    # 注意：这里我们只需要一个可调用对象让 Keras 能找到它，实际推理时不会计算损失。
    return tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y_true, y_pred))

# 如果需要 AdamW 优化器
from tensorflow.keras.optimizers import AdamW

ap = argparse.ArgumentParser()
ap.add_argument("-c", "--cascade", required=True, help="path to face cascade")
ap.add_argument("-m", "--model", required=True, help="path to emotion model")
ap.add_argument("-v", "--video", required=True, help="path to video file")
args = vars(ap.parse_args())

detector = cv2.CascadeClassifier(args["cascade"])
print("[INFO] loading model...")
# 加载模型时指定自定义对象
model = load_model(
    args["model"],
    custom_objects={
        'weighted_smooth_cce': weighted_smooth_cce,
        'AdamW': AdamW
    },
    safe_mode=False
)

EMOTIONS = ["angry", "fear", "happy", "sad", "surprise", "neutral"]

camera = cv2.VideoCapture(args["video"])

while True:
    (grabbed, frame) = camera.read()
    if not grabbed:
        break

    # ---------- 人脸检测部分（采用优化后的参数 + 几何过滤，保持检测稳定） ----------
    frame = imutils.resize(frame, width=300)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    canvas = np.zeros((420, 600, 3), dtype="uint8")
    frameClone = frame.copy()

    rects = detector.detectMultiScale(gray,
                                      scaleFactor=1.05,
                                      minNeighbors=8,
                                      minSize=(60, 60),
                                      flags=cv2.CASCADE_SCALE_IMAGE)

    if len(rects) > 0:
        # 几何过滤：宽高比
        rects = [r for r in rects if 0.8 <= r[2] / r[3] <= 1.2]
        area_frame = gray.shape[0] * gray.shape[1]
        rects = [r for r in rects if 0.005 < (r[2] * r[3]) / area_frame < 0.4]

        if len(rects) > 0:
            rect = sorted(rects, key=lambda x: x[2] * x[3], reverse=True)[0]
            (fx, fy, fw, fh) = rect

            roi = gray[fy:fy + fh, fx:fx + fw]
            roi = cv2.resize(roi, (48, 48))
            roi = roi.astype("float") / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            preds = model.predict(roi)[0]
            print("[INFO] ----", preds.argmax(), preds)
            label = EMOTIONS[preds.argmax()]

            # 绘制概率条
            for i, (emotion, prob) in enumerate(zip(EMOTIONS, preds)):
                text = "{}: {:.2f}%".format(emotion, prob * 100)
                w = int(prob * 600)
                y_top = i * 60 + 5
                y_bottom = y_top + 55
                cv2.rectangle(canvas, (5, y_top), (w, y_bottom), (0, 0, 255), -1)
                cv2.putText(canvas, text, (10, y_top + 38), cv2.FONT_HERSHEY_SIMPLEX,
                            1.0, (255, 255, 255), 2)

            cv2.putText(frameClone, label, (fx, fy - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 0, 255), 2)
            cv2.rectangle(frameClone, (fx, fy), (fx + fw, fy + fh), (0, 0, 255), 2)

    h, w = frameClone.shape[:2]
    frame_large = cv2.resize(frameClone, (600, int(600 * h / w)))
    combined = np.vstack([frame_large, canvas])
    cv2.imshow("Emotion Detection", combined)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()