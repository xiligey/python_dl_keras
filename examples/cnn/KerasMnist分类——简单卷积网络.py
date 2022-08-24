"""
设计一个简单的卷积神经网络：

1. 第一个隐藏层是一个卷积层，使用5*5的感受野，输出具有32个特征图，输入的数据期待具有input_shape参数所描述的特征，采用ReLU作为激活函数
2. 定义一个采用最大值的池化层，采样因子为2*2
3. Dropout正则化层，随机排除20%
4. 将多维数据转为一维的Flatten层
5. 具有128个神经元的全连接层，采用ReLU
6. 输出层具有10个神经元，采用softmax
"""
import numpy as np
from keras import backend
from keras.datasets import mnist
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.models import Sequential
from keras.utils import np_utils

backend.set_image_data_format('channels_first')

# 设定随机种子
seed = 7
np.random.seed(seed)

# 从Keras导入Mnist数据集
(X_train, y_train), (X_validation, y_validation) = mnist.load_data()

X_train = X_train.reshape(X_train.shape[0], 1, 28, 28).astype('float32')
X_validation = X_validation.reshape(X_validation.shape[0], 1, 28, 28).astype('float32')

# 格式化数据到0-1之前
X_train = X_train / 255
X_validation = X_validation / 255

# one-hot编码
y_train = np_utils.to_categorical(y_train)
y_validation = np_utils.to_categorical(y_validation)


# 定义cnn模型


def create_model():
    # 创建模型
    model = Sequential()
    model.add(Conv2D(32, (5, 5), input_shape=(1, 28, 28), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(units=128, activation='relu'))
    model.add(Dense(units=10, activation='softmax'))

    # 编译模型
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


model = create_model()
model.fit(X_train, y_train, epochs=10, batch_size=200, verbose=2)

score = model.evaluate(X_validation, y_validation, verbose=0)
print('CNN_small: %.2f%%' % (score[1] * 100))  # 99.05%
