import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn import datasets
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# 导入数据
dataset = datasets.load_boston()

x, Y = dataset.data, dataset.target

# 设定随机种子
seed = 7
np.random.seed(seed)


def create_model(units_list=(13,), optimizer='adam', init='normal'):
    # 构建模型
    model = Sequential()

    # 构建第一个隐藏层和输入层
    units = units_list[0]
    model.add(Dense(units=units, activation='relu', input_dim=13, kernel_initializer=init))
    # 构建更多隐藏层
    for units in units_list[1:]:
        model.add(Dense(units=units, activation='relu', kernel_initializer=init))

    model.add(Dense(units=1, kernel_initializer=init))

    # 编译模型
    model.compile(loss='mean_squared_error', optimizer=optimizer)

    return model


model = KerasRegressor(build_fn=create_model, epochs=200, batch_size=5, verbose=0)
# 设置算法评估基准
kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
grid = cross_val_score(model, x, Y, cv=kfold)
print('Baseline: %.2f (%.2f) MSE' % (grid.mean(), grid.std()))

# ---------------------------正则化--------------------------
pipeline = Pipeline([
    ('standardize', StandardScaler()),
    ('mlp', model)
])
kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
grid = cross_val_score(pipeline, x, Y, cv=kfold)
print('Standardize: %.2f (%.2f) MSE' % (grid.mean(), grid.std()))

# GridSearch
# 调参选择最优模型
param_grid = {
    'units_list': [(20,), (13, 6)],
    'optimizer': ['rmsprop', 'adam'],
    'init': ['glorot_uniform', 'normal'],
    'epochs': [100, 200],
    'batch_size': [5, 20]
}
scaler = StandardScaler()
scaler_x = scaler.fit_transform(x)
grid = GridSearchCV(estimator=model, param_grid=param_grid)
grid.fit(scaler_x, Y)
# 输出结果
print('Best: %f using %s' % (grid.best_score_, grid.best_params_))
means = grid.cv_results_['mean_test_score']
stds = grid.cv_results_['std_test_score']
params = grid.cv_results_['params']

for mean, std, param in zip(means, stds, params):
    print('%f (%f) with: %r' % (mean, std, param))
