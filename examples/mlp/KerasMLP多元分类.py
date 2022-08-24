import keras
import numpy as np
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.optimizers import SGD

# 生成虚拟数据
x_train = np.random.random((1000, 20))
y_train = keras.utils.to_categorical(np.random.randint(10, size=(1000, 1)), num_classes=10)
x_test = np.random.random((100, 20))
y_test = keras.utils.to_categorical(np.random.randint(10, size=(100, 1)), num_classes=10)

model = Sequential()
# Dense(64) 是一个具有 64 个隐藏神经元的全连接层。
# 在第一层必须指定所期望的输入数据尺寸：
# 在这里，是一个 20 维的向量。
model.add(Dense(64, activation='relu', input_dim=20))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

model.fit(x_train, y_train,
          epochs=20,
          batch_size=128)
score = model.evaluate(x_test, y_test, batch_size=128)

print(model.predict(x_train))
print(score)

"""
array([[0.10370545, 0.09403334, 0.10659299, ..., 0.07923749, 0.10641167,
        0.11288734],
       [0.09545048, 0.10190095, 0.10142185, ..., 0.09453631, 0.10654115,
        0.09460115],
       [0.09676434, 0.10113858, 0.10378075, ..., 0.09417275, 0.10299642,
        0.09779805],
       ...,
       [0.09532722, 0.11207749, 0.1015248 , ..., 0.09108935, 0.09979115,
        0.09456217],
       [0.10323501, 0.10472632, 0.10497884, ..., 0.09161669, 0.10147148,
        0.09592628],
       [0.10169027, 0.11125398, 0.10036505, ..., 0.09027231, 0.10005651,
        0.09770641]], dtype=float32)
[2.3069868087768555, 0.07999999821186066]
"""
