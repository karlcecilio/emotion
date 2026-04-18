#   USAGE
#   python train_recognizer.py --checkpoints ./datasets/fer2013/checkpoints --model_type emotion_vggnet
#   python train_recognizer.py --checkpoints ./datasets/fer2013/checkpoints --model_type simple_cnn
#   python train_recognizer.py --checkpoints ./datasets/fer2013/checkpoints --model_type mobilenetv2
#   python train_recognizer.py --checkpoints ./datasets/fer2013/checkpoints --model_type mini_xception_optimized3

import matplotlib
matplotlib.use('Agg')

from config import emotion_config as config
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from pyimage.model.emotionvggnet import EmotionVGGNet
from pyimage.model.simple_cnn import SimpleCNN
from pyimage.model.mini_xception import MiniXception
from pyimage.io.hdf5datasetgenerator import HDF5DatasetGenerator
from pyimage.preprocessing.preprocessor import ImageToArrayPreprocessor
from pyimage.callback.epochcheckpoint import EpochCheckpoint
from pyimage.callback.trainingmonitor import TrainingMonitor
from pyimage.model.mobilenetv2_fer import MobileNetV2FER
import argparse
import os
import h5py
import sys
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, LearningRateScheduler
import tensorflow as tf
from tensorflow.keras.losses import CategoricalCrossentropy
import math

ap = argparse.ArgumentParser()
ap.add_argument("-c", "--checkpoints", required=True, help="path to checkpoint directory")
ap.add_argument("-m", "--model", type=str, help="path to existing model (for resuming training)")
ap.add_argument("-s", "--start-epoch", type=int, default=0, help="epoch to restart training at")
ap.add_argument("-t", "--model_type", type=str, default="emotion_vggnet",
                choices=["simple_cnn", "emotion_vggnet", "mini_xception", "mobilenetv2",
                         "mini_xception_optimized1", "mini_xception_optimized2", "mini_xception_optimized3"],
                help="type of model to use")
args = vars(ap.parse_args())

# ======================== 辅助函数 ========================
def make_basic_callbacks(model_type, checkpoints, start_epoch, output_path):
    """生成基础回调（EpochCheckpoint + TrainingMonitor）"""
    figPath = os.path.join(output_path, f"{model_type}_training.png")
    jsonPath = os.path.join(output_path, f"{model_type}_history.json")
    callbacks = [
        EpochCheckpoint(checkpoints, every=10, startAt=start_epoch),
        TrainingMonitor(figPath=figPath, jsonPath=jsonPath, startAt=start_epoch)
    ]
    return callbacks

# 默认数据增强（可能被优化分支覆盖）
trainAug = ImageDataGenerator(rotation_range=10, zoom_range=0.1,
                              horizontal_flip=True, rescale=1./255, fill_mode='nearest')
valAug = ImageDataGenerator(rescale=1./255)
iap = ImageToArrayPreprocessor()

trainGen = HDF5DatasetGenerator(config.TRAIN_HDF5, config.BATCH_SIZE,
                                aug=trainAug, preprocessor=iap, classes=config.NUM_CLASSES)
valGen = HDF5DatasetGenerator(config.VAL_HDF5, config.BATCH_SIZE,
                              aug=valAug, preprocessor=iap, classes=config.NUM_CLASSES)

start_epoch = args["start_epoch"]

# ---------- 模型构建 ----------
if args["model"] is None:
    print("[INFO] compiling new model of type: {}".format(args["model_type"]))

    # -------------------- 基线模型 --------------------
    if args["model_type"] == "simple_cnn":
        model = SimpleCNN.build(width=48, height=48, depth=1, classes=config.NUM_CLASSES)
        opt = Adam(learning_rate=1e-3)
        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
        callbacks = make_basic_callbacks(args["model_type"], args["checkpoints"], start_epoch, config.OUTPUT_PATH)
        total_epochs = start_epoch + 60
        model.fit(
            trainGen.generator(),
            steps_per_epoch=trainGen.numImages // config.BATCH_SIZE,
            validation_data=valGen.generator(),
            validation_steps=valGen.numImages // config.BATCH_SIZE,
            callbacks=callbacks,
            epochs=total_epochs,
            initial_epoch=start_epoch,
            verbose=1,
        )
        trainGen.close()
        valGen.close()

    elif args["model_type"] == "emotion_vggnet":
        model = EmotionVGGNet.build(width=48, height=48, depth=1, classes=config.NUM_CLASSES)
        opt = Adam(learning_rate=1e-2)
        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
        callbacks = make_basic_callbacks(args["model_type"], args["checkpoints"], start_epoch, config.OUTPUT_PATH)
        total_epochs = start_epoch + 60
        model.fit(
            trainGen.generator(),
            steps_per_epoch=trainGen.numImages // config.BATCH_SIZE,
            validation_data=valGen.generator(),
            validation_steps=valGen.numImages // config.BATCH_SIZE,
            callbacks=callbacks,
            epochs=total_epochs,
            initial_epoch=start_epoch,
            verbose=1,
        )
        trainGen.close()
        valGen.close()

    elif args["model_type"] == "mini_xception":
        model = MiniXception.build(width=48, height=48, depth=1, classes=config.NUM_CLASSES)
        opt = Adam(learning_rate=1e-3)
        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
        callbacks = make_basic_callbacks(args["model_type"], args["checkpoints"], start_epoch, config.OUTPUT_PATH)
        total_epochs = start_epoch + 60
        model.fit(
            trainGen.generator(),
            steps_per_epoch=trainGen.numImages // config.BATCH_SIZE,
            validation_data=valGen.generator(),
            validation_steps=valGen.numImages // config.BATCH_SIZE,
            callbacks=callbacks,
            epochs=total_epochs,
            initial_epoch=start_epoch,
            verbose=1,
        )
        trainGen.close()
        valGen.close()

    elif args["model_type"] == "mobilenetv2":
        model = MobileNetV2FER.build(width=48, height=48, depth=1, classes=config.NUM_CLASSES)
        opt = Adam(learning_rate=1e-3)
        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
        callbacks = make_basic_callbacks(args["model_type"], args["checkpoints"], start_epoch, config.OUTPUT_PATH)
        total_epochs = start_epoch + 60
        model.fit(
            trainGen.generator(),
            steps_per_epoch=trainGen.numImages // config.BATCH_SIZE,
            validation_data=valGen.generator(),
            validation_steps=valGen.numImages // config.BATCH_SIZE,
            callbacks=callbacks,
            epochs=total_epochs,
            initial_epoch=start_epoch,
            verbose=1,
        )
        trainGen.close()
        valGen.close()

    # -------------------- 第一次优化 --------------------
    elif args["model_type"] == "mini_xception_optimized1":
        model = MiniXception.build(width=48, height=48, depth=1, classes=config.NUM_CLASSES)

        with h5py.File(config.TRAIN_HDF5, 'r') as f:
            train_labels = f['labels'][:]
        classes = np.unique(train_labels)
        class_weights = compute_class_weight('balanced', classes=classes, y=train_labels)
        class_weight_dict = dict(enumerate(class_weights))
        print("[INFO] Class weights:", class_weight_dict)

        class_weight_tensor = tf.constant([class_weight_dict[i] for i in range(config.NUM_CLASSES)], dtype=tf.float32)

        def weighted_cce(y_true, y_pred):
            y_true_int = tf.argmax(y_true, axis=1)
            weights = tf.gather(class_weight_tensor, y_true_int)
            cce = CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)(y_true, y_pred)
            return tf.reduce_mean(cce * weights)

        opt = Adam(learning_rate=1e-3)
        model.compile(loss=weighted_cce, optimizer=opt, metrics=['accuracy'])

        callbacks = make_basic_callbacks(args["model_type"], args["checkpoints"], start_epoch, config.OUTPUT_PATH)
        lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)
        early_stop = EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True, verbose=1)
        callbacks.extend([lr_scheduler, early_stop])

        total_epochs = start_epoch + 80
        history = model.fit(
            trainGen.generator(),
            steps_per_epoch=trainGen.numImages // config.BATCH_SIZE,
            validation_data=valGen.generator(),
            validation_steps=valGen.numImages // config.BATCH_SIZE,
            callbacks=callbacks,
            epochs=total_epochs,
            initial_epoch=start_epoch,
            verbose=1,
        )
        val_acc = history.history['val_accuracy']
        best_epoch = np.argmax(val_acc) + 1
        best_val_acc = val_acc[best_epoch - 1]
        print(f"[INFO] Best epoch: {best_epoch}, best val_accuracy: {best_val_acc:.4f}")
        best_model_path = os.path.join(args["checkpoints"], f"mini_xception_optimized1_best_epoch_{best_epoch}.hdf5")
        model.save(best_model_path)
        trainGen.close()
        valGen.close()

    # -------------------- 第二次优化 --------------------
    elif args["model_type"] == "mini_xception_optimized2":
        trainGen.close()
        valGen.close()

        trainAug = ImageDataGenerator(
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.15,
            horizontal_flip=True,
            brightness_range=[0.8, 1.2],
            rescale=1./255,
            fill_mode='nearest'
        )
        valAug = ImageDataGenerator(rescale=1./255)
        iap = ImageToArrayPreprocessor()

        trainGen = HDF5DatasetGenerator(config.TRAIN_HDF5, config.BATCH_SIZE,
                                        aug=trainAug, preprocessor=iap, classes=config.NUM_CLASSES)
        valGen = HDF5DatasetGenerator(config.VAL_HDF5, config.BATCH_SIZE,
                                      aug=valAug, preprocessor=iap, classes=config.NUM_CLASSES)

        model = MiniXception.build(width=48, height=48, depth=1, classes=config.NUM_CLASSES)

        with h5py.File(config.TRAIN_HDF5, 'r') as f:
            train_labels = f['labels'][:]
        classes = np.unique(train_labels)
        class_weights = compute_class_weight('balanced', classes=classes, y=train_labels)
        class_weight_dict = dict(enumerate(class_weights))
        class_weight_tensor = tf.constant([class_weight_dict[i] for i in range(config.NUM_CLASSES)], dtype=tf.float32)

        def weighted_cce(y_true, y_pred):
            y_true_int = tf.argmax(y_true, axis=1)
            weights = tf.gather(class_weight_tensor, y_true_int)
            cce = CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)(y_true, y_pred)
            return tf.reduce_mean(cce * weights)

        opt = Adam(learning_rate=1e-3)
        model.compile(loss=weighted_cce, optimizer=opt, metrics=['accuracy'])

        callbacks = make_basic_callbacks(args["model_type"], args["checkpoints"], start_epoch, config.OUTPUT_PATH)
        lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)
        early_stop = EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True, verbose=1)
        callbacks.extend([lr_scheduler, early_stop])

        total_epochs = start_epoch + 80
        history = model.fit(
            trainGen.generator(),
            steps_per_epoch=trainGen.numImages // config.BATCH_SIZE,
            validation_data=valGen.generator(),
            validation_steps=valGen.numImages // config.BATCH_SIZE,
            callbacks=callbacks,
            epochs=total_epochs,
            initial_epoch=start_epoch,
            verbose=1,
        )
        val_acc = history.history['val_accuracy']
        best_epoch = np.argmax(val_acc) + 1
        best_val_acc = val_acc[best_epoch - 1]
        print(f"[INFO] Best epoch: {best_epoch}, best val_accuracy: {best_val_acc:.4f}")
        best_model_path = os.path.join(args["checkpoints"], f"mini_xception_optimized2_best_epoch_{best_epoch}.hdf5")
        model.save(best_model_path)
        trainGen.close()
        valGen.close()

    # -------------------- 第三次优化：中等容量模型 + 标签平滑 + MixUp + 余弦退火 --------------------
    elif args["model_type"] == "mini_xception_optimized3":
        from pyimage.model.medium_mini_xception import MediumMiniXception

        trainGen.close()
        valGen.close()

        # 温和数据增强（无亮度变化，避免破坏小图）
        trainAug = ImageDataGenerator(
            rotation_range=10,
            width_shift_range=0.05,
            height_shift_range=0.05,
            zoom_range=0.1,
            horizontal_flip=True,
            rescale=1./255,
            fill_mode='nearest'
        )
        valAug = ImageDataGenerator(rescale=1./255)
        iap = ImageToArrayPreprocessor()

        trainGen = HDF5DatasetGenerator(config.TRAIN_HDF5, config.BATCH_SIZE,
                                        aug=trainAug, preprocessor=iap, classes=config.NUM_CLASSES)
        valGen = HDF5DatasetGenerator(config.VAL_HDF5, config.BATCH_SIZE,
                                      aug=valAug, preprocessor=iap, classes=config.NUM_CLASSES)

        model = MediumMiniXception.build(width=48, height=48, depth=1, classes=config.NUM_CLASSES)
        print("\n=== Model Summary ===")
        model.summary()
        print(f"Total parameters: {model.count_params():,}\n")

        # 类别权重
        with h5py.File(config.TRAIN_HDF5, 'r') as f:
            train_labels = f['labels'][:]
        classes = np.unique(train_labels)
        class_weights = compute_class_weight('balanced', classes=classes, y=train_labels)
        class_weight_dict = dict(enumerate(class_weights))
        print("[INFO] Class weights:", class_weight_dict)

        class_weight_tensor = tf.constant([class_weight_dict[i] for i in range(config.NUM_CLASSES)], dtype=tf.float32)
        smooth_factor = 0.1

        def weighted_smooth_cce(y_true, y_pred):
            num_classes = tf.shape(y_true)[-1]
            y_true_smooth = y_true * (1 - smooth_factor) + (smooth_factor / tf.cast(num_classes, tf.float32))
            y_true_int = tf.argmax(y_true, axis=1)
            weights = tf.gather(class_weight_tensor, y_true_int)
            cce = CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)(y_true_smooth, y_pred)
            return tf.reduce_mean(cce * weights)

        opt = Adam(learning_rate=0.001)
        model.compile(loss=weighted_smooth_cce, optimizer=opt, metrics=['accuracy'])

        # 余弦退火学习率
        total_epochs = 100
        def cosine_decay(epoch):
            lr = 0.001 * (1 + math.cos(math.pi * epoch / total_epochs)) / 2
            return max(lr, 1e-6)
        lr_scheduler = LearningRateScheduler(cosine_decay, verbose=1)
        early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1)

        # MixUp 生成器
        def mixup_generator(gen, alpha=0.2):
            while True:
                images, labels = next(gen)
                batch_size = images.shape[0]  # 获取当前批次实际大小
                indices = np.random.permutation(batch_size)
                mixed_images = []
                mixed_labels = []
                for i in range(batch_size):
                    lam = np.random.beta(alpha, alpha)
                    j = indices[i]
                    mixed_img = lam * images[i] + (1 - lam) * images[j]
                    mixed_label = lam * labels[i] + (1 - lam) * labels[j]
                    mixed_images.append(mixed_img)
                    mixed_labels.append(mixed_label)
                yield np.array(mixed_images), np.array(mixed_labels)

        callbacks = make_basic_callbacks(args["model_type"], args["checkpoints"], start_epoch, config.OUTPUT_PATH)
        callbacks.extend([lr_scheduler, early_stop])

        history = model.fit(
            mixup_generator(trainGen.generator()),
            steps_per_epoch=trainGen.numImages // config.BATCH_SIZE,
            validation_data=valGen.generator(),
            validation_steps=valGen.numImages // config.BATCH_SIZE,
            callbacks=callbacks,
            epochs=total_epochs,
            initial_epoch=start_epoch,
            verbose=1,
        )

        val_acc = history.history['val_accuracy']
        best_epoch = np.argmax(val_acc) + 1
        best_val_acc = val_acc[best_epoch - 1]
        print(f"[INFO] Best epoch: {best_epoch}, best val_accuracy: {best_val_acc:.4f}")
        best_model_path = os.path.join(args["checkpoints"], f"mini_xception_optimized3_best_epoch_{best_epoch}.hdf5")
        model.save(best_model_path)
        trainGen.close()
        valGen.close()

    else:
        raise ValueError("Unsupported model type")

else:
    # 恢复训练（仅适用于基线模型，优化分支不支持恢复）
    print("[INFO] loading existing model from {}".format(args["model"]))
    model = load_model(args["model"])
    new_lr = 1e-4
    opt = Adam(learning_rate=new_lr)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    print("[INFO] recompiled model with learning rate = {}".format(new_lr))

    callbacks = make_basic_callbacks(args["model_type"], args["checkpoints"], start_epoch, config.OUTPUT_PATH)
    total_epochs = start_epoch + 60
    model.fit(
        trainGen.generator(),
        steps_per_epoch=trainGen.numImages // config.BATCH_SIZE,
        validation_data=valGen.generator(),
        validation_steps=valGen.numImages // config.BATCH_SIZE,
        callbacks=callbacks,
        epochs=total_epochs,
        initial_epoch=start_epoch,
        verbose=1,
    )
    trainGen.close()
    valGen.close()