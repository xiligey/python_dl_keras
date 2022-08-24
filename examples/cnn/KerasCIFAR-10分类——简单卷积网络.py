"""
1. 卷积层，32个特征图，感受野3*3
2. Dropout层，20%
3. 卷积层，32个特征图，感受野3*3
4. Dropout，20%
5. 池化层，2*2
6. Flatten层
7. 512个神经元的全连接层，激活函数为ReLU
8. Dropout，50%
9. 10个神经元的输出层，激活函数为softmax
"""
import matplotlib.pyplot as plt
import numpy as np
from keras import backend
from keras.constraints import maxnorm
from keras.datasets import cifar10
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.models import Sequential
from keras.optimizers import SGD
from keras.utils import np_utils
from PIL import Image

backend.set_image_data_format('channels_first')

(X_train, y_train), (X_validation, y_validation) = cifar10.load_data()
for i in range(0, 9):
    plt.subplot(331 + i)
    plt.imshow(Image.fromarray(X_train[i]))

# 显示图片
plt.show()

# 设定随机种子
seed = 7
np.random.seed(seed)
# 格式化数据到0-1之前
X_train = X_train.astype('float32')
X_validation = X_validation.astype('float32')
X_train = X_train / 255.0
X_validation = X_validation / 255.0

# one-hot编码
y_train = np_utils.to_categorical(y_train)
y_validation = np_utils.to_categorical(y_validation)
num_classes = y_train.shape[1]


def create_model(epochs=25):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(3, 32, 32),
                     padding='same', activation='relu', kernel_constraint=maxnorm(3)))
    model.add(Dropout(0.2))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', kernel_constraint=maxnorm(3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(512, activation='relu', kernel_constraint=maxnorm(3)))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))
    lrate = 0.01
    decay = lrate / epochs
    sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return model


epochs = 25
model = create_model(epochs)
model.fit(x=X_train, y=y_train, epochs=epochs, batch_size=32, verbose=2)
scores = model.evaluate(x=X_validation, y=y_validation, verbose=0)
print('Accuracy: %.2f%%' % (scores[1] * 100))  # 70.34%
