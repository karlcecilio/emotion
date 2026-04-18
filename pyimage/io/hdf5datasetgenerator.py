import h5py
import numpy as np
from tensorflow.keras.utils import to_categorical

class HDF5DatasetGenerator:
    def __init__(self, dbPath, batchSize, aug=None, preprocessor=None, classes=None):
        self.db = h5py.File(dbPath, 'r')
        self.batchSize = batchSize
        self.aug = aug
        self.preprocessor = preprocessor
        self.classes = classes
        # 关键修改：数据集名称改为 'image'（与 HDF5DatasetWriter 的默认值一致）
        self.numImages = self.db['image'].shape[0]

    def generator(self, passes=np.inf):
        epochs = 0
        while epochs < passes:
            for i in range(0, self.numImages, self.batchSize):
                images = self.db['image'][i:i + self.batchSize]
                labels = self.db['labels'][i:i + self.batchSize]

                # 预处理：如果提供了预处理器
                if self.preprocessor is not None:
                    procImages = []
                    for img in images:
                        # 根据存储形状调整：如果图像被展平（一维），则重塑为 (48,48,1)
                        if img.ndim == 1:
                            img = img.reshape(48, 48, 1)
                        elif img.ndim == 2:
                            # 若为2D灰度图，增加通道维度
                            img = np.expand_dims(img, axis=-1)
                        # 调用预处理器（例如 ImageToArrayPreprocessor）
                        img = self.preprocessor.preprocess(img)
                        procImages.append(img)
                    images = np.array(procImages)

                # 数据增强
                if self.aug:
                    images, labels = next(self.aug.flow(images, labels, batch_size=self.batchSize))

                # 标签 one-hot 编码
                if self.classes is not None:
                    labels = to_categorical(labels, num_classes=self.classes)

                yield (images, labels)
            epochs += 1

    def close(self):
        self.db.close()