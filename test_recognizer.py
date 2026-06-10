#   USAGE
#   python test_recognizer.py --model ./datasets/fer2013/checkpoints/mini_xception_final_best_epoch_106.hdf5

import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.losses import CategoricalCrossentropy
from config import emotion_config as config
from pyimage.preprocessing.preprocessor import ImageToArrayPreprocessor
from pyimage.io.hdf5datasetgenerator import HDF5DatasetGenerator

def weighted_smooth_cce(y_true, y_pred):
    smooth_factor = 0.1
    num_classes = tf.shape(y_true)[-1]
    y_true_smooth = y_true * (1 - smooth_factor) + (smooth_factor / tf.cast(num_classes, tf.float32))
    cce = CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)(y_true_smooth, y_pred)
    return tf.reduce_mean(cce)

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", type=str, required=True, help="path to model checkpoint to load")
args = vars(ap.parse_args())

print("[INFO] loading {}...".format(args["model"]))
model = load_model(
    args["model"],
    custom_objects={
        'weighted_smooth_cce': weighted_smooth_cce,
        'AdamW': tf.keras.optimizers.AdamW
    }
)

# 创建测试数据生成器
testAug = ImageDataGenerator(rescale=1. / 255)
iap = ImageToArrayPreprocessor()

testGen = HDF5DatasetGenerator(
    config.TEST_HDF5,
    config.BATCH_SIZE,
    aug=testAug,
    preprocessor=iap,
    classes=config.NUM_CLASSES
)

# 手动遍历所有批次，收集真实标签和预测标签
print("[INFO] collecting predictions for confusion matrix...")
steps = testGen.numImages // config.BATCH_SIZE
y_true = []
y_pred = []

for i in range(steps):
    images, labels = next(testGen.generator())
    preds = model.predict(images, verbose=0)
    y_true.extend(np.argmax(labels, axis=1))
    y_pred.extend(np.argmax(preds, axis=1))

# 计算准确率
acc = np.mean(np.array(y_true) == np.array(y_pred))
print("[INFO] test accuracy: {:.4f} ({:.2f}%)".format(acc, acc*100))

# 混淆矩阵
cm = confusion_matrix(y_true, y_pred)
print("[INFO] confusion matrix:")
print(cm)

# 分类报告
class_names = ['Angry/Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
print("\n[INFO] classification report:")
print(classification_report(y_true, y_pred, target_names=class_names))

# 绘制混淆矩阵热力图
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix on Test Set')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300)
print("[INFO] confusion matrix saved as confusion_matrix.png")

testGen.close()