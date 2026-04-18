import matplotlib.pyplot as plt
import json
import os
from tensorflow.keras.callbacks import Callback

class TrainingMonitor(Callback):
    def __init__(self, figPath, jsonPath=None, startAt=0):
        super(Callback, self).__init__()
        self.figPath = figPath
        self.jsonPath = jsonPath
        self.startAt = startAt
        self.epochs = []
        self.history = {}

    def on_train_begin(self, logs=None):
        if self.jsonPath is not None and os.path.exists(self.jsonPath):
            with open(self.jsonPath, 'r') as f:
                self.history = json.load(f)
            # 只保留 startAt 之前的历史数据
            for k in self.history:
                self.history[k] = self.history[k][:self.startAt]
            # 【修复】根据保留的历史数据长度初始化 epochs 列表
            if 'loss' in self.history:
                self.epochs = list(range(1, len(self.history['loss']) + 1))
            else:
                self.epochs = []

    def on_epoch_end(self, epoch, logs=None):
        epoch += 1  # 转为从1开始
        self.epochs.append(epoch)

        for k, v in logs.items():
            if k not in self.history:
                self.history[k] = []
            self.history[k].append(v)

        if self.jsonPath is not None:
            os.makedirs(os.path.dirname(self.jsonPath), exist_ok=True)
            with open(self.jsonPath, 'w') as f:
                json.dump(self.history, f)

        # 【可选】防御：如果长度不一致，重新对齐 epochs
        if len(self.epochs) != len(self.history.get('loss', [])):
            self.epochs = list(range(1, len(self.history['loss']) + 1))

        plt.style.use("ggplot")
        plt.figure()

        if 'loss' in self.history and 'val_loss' in self.history:
            plt.plot(self.epochs, self.history['loss'], 'b-', label='train_loss')
            plt.plot(self.epochs, self.history['val_loss'], 'r-', label='val_loss')

        if 'accuracy' in self.history and 'val_accuracy' in self.history:
            plt.plot(self.epochs, self.history['accuracy'], 'g-', label='train_acc')
            plt.plot(self.epochs, self.history['val_accuracy'], 'k-', label='val_acc')
        elif 'acc' in self.history and 'val_acc' in self.history:
            plt.plot(self.epochs, self.history['acc'], 'g-', label='train_acc')
            plt.plot(self.epochs, self.history['val_acc'], 'k-', label='val_acc')

        plt.xlabel("Epoch")
        plt.ylabel("Loss / Accuracy")
        plt.legend()
        os.makedirs(os.path.dirname(self.figPath), exist_ok=True)
        plt.savefig(self.figPath)
        plt.close()