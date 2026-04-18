# ImageToArrayPreprocessor.py
from tensorflow.keras.preprocessing.image import img_to_array


class ImageToArrayPreprocessor:
    def __init__(self, dataFormat=None):
        """
        初始化预处理器

        参数:
        - dataFormat: 图像通道顺序，'channels_first' 或 'channels_last'，默认为 None（由 Keras 自动决定）
        """
        self.dataFormat = dataFormat

    def preprocess(self, image):
        """
        将图像转换为 Keras 可用的数组格式

        参数:
        - image: 输入图像（numpy 数组，通常形状为 (height, width) 或 (height, width, channels)）

        返回:
        - 经过 img_to_array 转换后的数组
        """
        return img_to_array(image, data_format=self.dataFormat)