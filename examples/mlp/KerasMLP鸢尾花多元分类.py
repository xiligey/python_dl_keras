from sklearn.datasets import load_iris
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

seed = 7
np.random.seed(seed)

# 导入数据
dataset = load_iris()
x, y = dataset.data, dataset.target


def create_model(optimizer='adam', init='glorot_uniform'):
    model = Sequential()
    model.add(Dense(units=4, input_dim=4, activation='relu', kernel_initializer=init))
    model.add(Dense(6, activation='relu', kernel_initializer=init))
    model.add(Dense(3, activation='softmax', kernel_initializer=init))
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# 创建模型
model = KerasClassifier(build_fn=create_model, epochs=200, batch_size=5, verbose=0)

# 交叉验证
kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(model, x, y, cv=kfold)
print(results.mean())  # 0.9000000014901162
