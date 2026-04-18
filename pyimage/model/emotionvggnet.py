from keras.src.layers import Activation
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import ELU
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input

class EmotionVGGNet:
    @staticmethod
    def build(width, height, depth, classes):
        model = Sequential()
        inputShape = (height, width, depth)
        chanDim = -1
        if K.image_data_format() == 'channels_first':
            inputShape = (depth, height, width)
            chanDim = 1
        # Block1  第一块卷积核
        model.add(Input(shape=inputShape))
        model.add(Conv2D(32, (3, 3), padding='same', kernel_initializer='he_normal'))
        # model.add(Conv2D(32, (3, 3), padding='same',kernel_initializer='he_normal',input_shape=inputShape))
        model.add(ELU())
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(32, (3, 3), padding='same',kernel_initializer='he_normal'))
        model.add(ELU())
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        # Block2
        model.add(Conv2D(64, (3, 3), padding='same',kernel_initializer='he_normal'))
        model.add(ELU())
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(64, (3, 3), padding='same',kernel_initializer='he_normal'))
        model.add(ELU())
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        # Block3
        model.add(Conv2D(128, (3, 3), padding='same',kernel_initializer='he_normal'))
        model.add(ELU())
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(128, (3, 3), padding='same',kernel_initializer='he_normal'))
        model.add(ELU())
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        #Block4
        model.add(Flatten())
        model.add(Dense(64, kernel_initializer='he_normal'))
        model.add(ELU())
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

        #Block6
        model.add(Dense(64, kernel_initializer='he_normal'))
        model.add(ELU())
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

        #Block7
        model.add(Dense(classes,kernel_initializer='he_normal'))
        model.add(Activation('softmax'))

        return model
