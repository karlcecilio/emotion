import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input, Lambda, Resizing


class MobileNetV2FER:
    @staticmethod
    def build(width=48, height=48, depth=1, classes=6):
        input_tensor = Input(shape=(height, width, depth))
        # 1. 将 48x48 放大到 224x224（双线性插值）
        x = Resizing(224, 224, interpolation='bilinear')(input_tensor)
        # 2. 灰度转 RGB（复制三份）
        if depth == 1:
            x = Lambda(lambda img: tf.image.grayscale_to_rgb(img))(x)
        else:
            x = x
        # 3. 应用 MobileNetV2 的预处理器（将 [0,255] 转为 [-1,1]，注意输入此时是 [0,255] 范围）
        #    注意：我们的图像在生成器中已经除以255变成[0,1]，需要先还原为[0,255]？更简单：直接自定义预处理
        #    实际上 preprocess_input 期望输入 [0,255]，但我们已经做了 rescale=1/255。解决方案：取消 rescale 或者在这里反向操作。
        #    为简单，我们不使用 preprocess_input，手动做归一化到 [-1,1]
        #    因为从 [0,1] 转到 [-1,1]： x = x * 2 - 1
        x = Lambda(lambda img: img * 2.0 - 1.0)(x)  # 范围 [-1,1]

        base_model = MobileNetV2(weights='imagenet', include_top=False, input_tensor=x)
        base_model.trainable = False
        x = GlobalAveragePooling2D()(base_model.output)
        x = Dense(128, activation='relu')(x)
        output = Dense(classes, activation='softmax')(x)
        model = Model(inputs=input_tensor, outputs=output)
        return model