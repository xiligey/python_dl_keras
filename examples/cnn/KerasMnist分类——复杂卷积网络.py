"""
1. 卷积层，具有30个特征图，感受野5*5
2. 采样因子为2*2的池化层
3. 卷积层，具有15个特征图，感受野3*3
4. 采样因子为2*2的池化层
5. Dropout为0.2
6. Flatten层
7. 具有128个神经元和ReLU激活函数的全连接层
8. 具有50个神经元和ReLU激活函数的全连接层
9. 输出层
"""
from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense


def create_model():
    model = Sequential()
    model.add(Conv2D(30, (5, 5), input_shape=(1, 28, 28), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(15, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(units=128, activation='relu'))
    model.add(Dense(units=50, activation='relu'))
    model.add(Dense(units=10, activation='softmax'))

    # 编译模型
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
