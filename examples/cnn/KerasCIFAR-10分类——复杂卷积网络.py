"""
1. 卷积层，32个特征图，感受野3*3
2. Dropout层，20%
3. 卷积层，32，3*3
4. 池化，2*2
5. 卷积，64，3*3
6. Dropout，20%
7. 卷积，64，3*3
8. 池化，2*2
9. 卷积，128，3*3
10. Dropout，20%
11. 卷积，128，3*3
12. 池化，2*2
13. Flatten
14. Dropout，20%
15. 1024个神经元和ReLU激活函数的全连接层
16. Dropout，20%
17. 512个神经元和ReLU的全连接层
18. Dropout，20%
19. 10个神经元的输出层，激活函数为softmax
"""
from keras import Sequential
from keras.constraints import maxnorm
from keras.layers import Conv2D, Dropout, MaxPooling2D, Flatten, Dense
from keras.optimizers import SGD


def create_model(epochs=25):
    model = Sequential()
    model.add(
        Conv2D(32, (3, 3), input_shape=(3, 32, 32), padding='same', activation='relu', kernel_constraint=maxnorm(3)))
    model.add(Dropout(0.2))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', kernel_constraint=maxnorm(3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', kernel_constraint=maxnorm(3)))
    model.add(Dropout(0.2))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', kernel_constraint=maxnorm(3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same', kernel_constraint=maxnorm(3)))
    model.add(Dropout(0.2))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same', kernel_constraint=maxnorm(3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(1024, activation='relu', kernel_constraint=maxnorm(3)))
    model.add(Dropout(0.2))
    model.add(Dense(512, activation='relu', kernel_constraint=maxnorm(3)))
    model.add(Dropout(0.2))
    model.add(Dense(10, activation='softmax'))
    lrate = 0.01
    decay = lrate / epochs
    sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return model
# 准确率75.75%
