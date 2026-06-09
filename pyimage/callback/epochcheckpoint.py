# EpochCheckpoint.py
import os
from tensorflow.keras.callbacks import Callback


class EpochCheckpoint(Callback):
    def __init__(self, outputPath, every=20, startAt=0,model_type = "test"):
        """
        每个 epoch 结束后保存模型权重

        参数:
        - outputPath: 保存模型文件的目录
        - every: 每隔多少个 epoch 保存一次
        - startAt: 从哪个 epoch 编号开始记录（用于恢复训练）
        """
        super(Callback, self).__init__()
        self.outputPath = outputPath
        self.every = every
        self.startAt = startAt
        self.model_type = model_type

    def on_epoch_end(self, epoch, logs=None):
        # 将 epoch 转换为从 1 开始计数，与用户习惯一致
        epoch += 1
        if (epoch - self.startAt) % self.every == 0:
            # 创建保存路径（如果目录不存在则自动创建）
            os.makedirs(self.outputPath, exist_ok=True)
            path = os.path.join(self.outputPath, f"{self.model_type}_epoch_{epoch}.hdf5")
            self.model.save(path)
            print(f"[INFO] Epoch checkpoint saved to {path}")