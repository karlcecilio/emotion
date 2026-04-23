from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, SeparableConv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Dropout, BatchNormalization, Activation

class FinalMiniXception:
    @staticmethod
    def build(width, height, depth, classes):
        model = Sequential()
        # Block 1 (通道数适度增加)
        model.add(Conv2D(64, (3,3), padding='same', input_shape=(height,width,depth)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Conv2D(128, (3,3), padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D((2,2)))
        model.add(Dropout(0.25))

        # Block 2
        model.add(SeparableConv2D(256, (3,3), padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(SeparableConv2D(256, (3,3), padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D((2,2)))
        model.add(Dropout(0.25))

        # Block 3
        model.add(SeparableConv2D(512, (3,3), padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(SeparableConv2D(512, (3,3), padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D((2,2)))
        model.add(Dropout(0.25))

        # Global Average Pooling
        model.add(GlobalAveragePooling2D())
        model.add(Dense(512))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(classes, activation='softmax'))
        return model