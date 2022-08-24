"""
使用IMDB提供的数据及中的评论信息来分析一部电影的好坏
"""
from matplotlib import pyplot as plt
from keras.datasets import imdb
import numpy as np
from keras.preprocessing import sequence
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.layers import Dense, Flatten, Embedding
from keras.models import Sequential

(x_train, y_train), (x_validation, y_validation) = imdb.load_data()

# 合并训练集和评估数据集
x = np.concatenate((x_train, x_validation), axis=0)
y = np.concatenate((y_train, y_validation), axis=0)

print('x shape is %s, y shape is %s' % (x.shape, y.shape))
print('Classes: %s' % np.unique(y))

print('Total words: %s' % len(np.unique(np.hstack(x))))

result = [len(word) for word in x]
print('Mean: %.2f words (STD: %.2f)' % (np.mean(result), np.std(result)))

# 图表展示
plt.subplot(121)
plt.boxplot(result)
plt.subplot(122)
plt.hist(result)
plt.show()

"""
Classes: [0 1]
Total words: 88585
Mean: 234.76 words (STD: 172.91)
"""

# 文本处理采用词嵌入技术


seed = 7
top_words = 5000
max_words = 500
out_dimension = 32
batch_size = 128
epochs = 2


def create_model():
    model = Sequential()
    # 构建嵌入层
    model.add(Embedding(top_words, out_dimension, input_length=max_words))
    # 1维度卷积层
    model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(250, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model


np.random.seed(seed=seed)
# 导入数据
(x_train, y_train), (x_validation, y_validation) = imdb.load_data(num_words=top_words)
# 限定数据集的长度
x_train = sequence.pad_sequences(x_train, maxlen=max_words)
x_validation = sequence.pad_sequences(x_validation, maxlen=max_words)

# 生成模型
model = create_model()
model.fit(x_train, y_train, validation_data=(x_validation, y_validation),
          batch_size=batch_size, epochs=epochs, verbose=2)

# 准确率 88.66%
