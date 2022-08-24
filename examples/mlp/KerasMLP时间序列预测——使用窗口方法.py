"""
可以使用多个最近的时间项来进行下一个时间项的预测，这个方法被称为窗口方法。可以针对不同的问题，对窗口的大小进行调整。

例如：给定当前时间t，预测序列t+1中的值时，使用当前时间t以及前两个时间点t-1和t-2，因此，输入变量为t-2,t-1,t，输出变量为t+1
"""
import math

import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from matplotlib import pyplot as plt
from pandas import read_csv

seed = 7
batch_size = 2
epochs = 400
filename = 'international-airline-passengers.csv'
footer = 3
look_back = 3


def create_dataset(dataset):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        x = dataset[i: i + look_back, 0]
        dataX.append(x)
        y = dataset[i + look_back, 0]
        dataY.append(y)
        print('X: %s, Y: %s' % (x, y))
    return np.array(dataX), np.array(dataY)


def build_model():
    model = Sequential()
    model.add(Dense(units=12, input_dim=look_back, activation='relu'))
    model.add(Dense(units=8, activation='relu'))
    model.add(Dense(units=1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


if __name__ == '__main__':
    # 设置随机种子
    np.random.seed(seed)

    # 导入数据
    data = read_csv(filename, usecols=[1], engine='python', skipfooter=footer)
    dataset = data.values.astype('float32')
    train_size = int(len(dataset) * 0.67)
    validation_size = len(dataset) - train_size
    train, validation = dataset[0: train_size, :], dataset[train_size: len(dataset), :]

    # 创建dataset，让数据产生相关性
    X_train, y_train = create_dataset(train)
    X_validation, y_validation = create_dataset(validation)

    # 训练模型
    model = build_model()
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=2)

    # 评估模型
    train_score = model.evaluate(X_train, y_train, verbose=0)
    print('Train Score: %.2f MSE (%.2f RMSE)' % (train_score, math.sqrt(train_score)))
    validation_score = model.evaluate(X_validation, y_validation, verbose=0)
    print('Validatin Score: %.2f MSE (%.2f RMSE)' % (validation_score, math.sqrt(validation_score)))

    # 图表查看预测趋势
    predict_train = model.predict(X_train)
    predict_validation = model.predict(X_validation)

    # 构建通过训练集进行预测的图表数据
    predict_train_plot = np.empty_like(dataset)
    predict_train_plot[:, :] = np.nan
    predict_train_plot[look_back:len(predict_train) + look_back, :] = predict_train

    # 构建通过评估数据集进行预测的图表数据
    predict_validation_plot = np.empty_like(dataset)
    predict_validation_plot[:, :] = np.nan
    predict_validation_plot[len(predict_train) + look_back * 2 + 1:len(dataset) - 1, :] = predict_validation

    # 图表显示
    plt.plot(dataset, color='blue')
    plt.plot(predict_train_plot, color='green')
    plt.plot(predict_validation_plot, color='red')
    plt.show()
"""
Train Score: 501.07 MSE (22.38 RMSE)
Validatin Score: 2319.98 MSE (48.17 RMSE)
"""
