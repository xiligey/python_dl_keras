# Keras 的核心数据结构是 model，一种组织网络层的方式。最简单的模型是 Sequential 顺序模型，它由多个网络层线性堆叠。
import keras
import numpy as np
from keras.layers import Dense
from keras.models import Sequential

model = Sequential()
# 简单的使用add()来堆叠模型
model.add(Dense(units=64, activation='relu', input_dim=100))
model.add(Dense(units=10, activation='softmax'))

# 完成构建模型后，需要使用compile()来配置学习过程
model.compile(
    loss='categorical_crossentropy',
    optimizer='sgd',
    metrics=['accuracy']
)
# 如果需要，你还可以进一步配置你的优化器
# model.compile(
#     loss=keras.losses.categorical_crossentropy,
#     optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True)
# )

# 现在，可以批量的训练数据了
X_train = np.random.randn(1000, 100)
y_train = np.random.randn(1000, 10)
model.fit(X_train, y_train, epochs=5, batch_size=32)

# 评估模型性能
X_test = np.random.randn(300, 100)
y_test = np.random.randn(300, 10)
loss_and_metrics = model.evaluate(X_test, y_test, batch_size=128)
print('loss and metrics:', loss_and_metrics)

# 预测新的数据
X_pred = np.random.randn(200, 100)
y_pred = model.predict(X_pred, batch_size=128)
print('y_pred:', y_pred)
