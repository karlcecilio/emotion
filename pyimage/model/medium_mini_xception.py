from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, SeparableConv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Dropout, BatchNormalization, Activation

class MediumMiniXception:
    @staticmethod
    def build(width, height, depth, classes):
        model = Sequential()
        # Block 1
        model.add(Conv2D(48, (3,3), padding='same', input_shape=(height,width,depth)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Conv2D(96, (3,3), padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D((2,2)))
        model.add(Dropout(0.25))

        # Block 2
        model.add(SeparableConv2D(192, (3,3), padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(SeparableConv2D(192, (3,3), padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D((2,2)))
        model.add(Dropout(0.25))

        # Block 3
        model.add(SeparableConv2D(384, (3,3), padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(SeparableConv2D(384, (3,3), padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D((2,2)))
        model.add(Dropout(0.25))

        # Global Average Pooling 替代 Flatten
        model.add(GlobalAveragePooling2D())
        model.add(Dense(256))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(classes, activation='softmax'))
        return model