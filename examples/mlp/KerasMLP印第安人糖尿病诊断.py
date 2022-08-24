"""
在Keras 中的模型被定义为层序列。 本例创建一个序贯模型，并一次添加一个图层，

直到对网络拓扑满意为止：

- 首先要确保输入层具有正确的输入维度，使用 input_dim 参数创建第一层，并将其设置为 8，表示输入层有 8个输入变量，这与数据的维度一致
- 如何设定网络的层数及其类型？这是个非常困难的问题。 寻找最优的网络拓构是一个试错的过程，通过进行一系列的试验， 对找到最好的网络结构有非常好的启发作用。
  一般来说 需要一个足够大的网络来捕获问题的结构。在这个例子中， 为了简化这个过程 使用三层完全连接的网络结构。
- Keras通常使用 Dense 类来定义完全连接的层 将层中的神经元数量unit指定为第一个参数，初始化方法（ init ）作为第二个参数，并使用 activation 参数指定激活函数。
  通常，将网络权重初始化为均匀分布的小随机数（ uniform ），在这个例子中使用介于 05 的随机数，这是 Keras 中的默认均衡权重初始化数值， 可以使用高斯分布产生的小随机数
- 使用 ReLU 作为前两层的激活函数，使用 sigmoid 作为输出层的激活函数，通常采用sigmoid tanh 作为激活函数 这是构建所有层的首选。
  现在的研究表明 使用 ReLU激活函数 ，可以得到更好的性能。 二分类的 输出层通常采用 sigmoid 作为激活函数，
 因此在这个例子中采用 sigmoid 作为输出层的激活函数 通过 Sequential add函数将层添加到模型，
 并组合在一起 在这个例子 ，第一个隐藏层有 12个 神经元 使用8个输入变量。 第二个 隐藏层有8 个神经元 最后输出层有1 个神经元来预测数据结果（是否患有糖尿病）
- 网络结构为：可视层(8个输入) --> 隐藏层(12个神经元) --> 隐藏层(8个神经元) --> 输出层(1个输出)

模型定义完后，需要编译，编译是为了使模型能够有效地使用Keras封装的数值计算。Keras可以根据后端自动选择最佳方法来训练模型并预测。

编译时，必须指定评估一组权重的损失函数loss、用于搜索网络中不同权重的优化器optimizer以及希望在模型训练期间收集和报告的评估指标。

在Keras中，二元分类问题的损失函数为二元交叉熵。使用有效地梯度下降算法Adam作为优化器，这是一个有效地默认值。使用准确率来衡量模型。
"""
import numpy as np
from keras.layers import Dense
from keras.models import Sequential

np.random.seed(1234)

# 加载csv数据
csv = './pima-indians-diabetes.csv'
array = np.loadtxt(csv, delimiter=',')
# 分离输入变量和输出变量
X, Y = array[:, :8], array[:, 8]
# 创建模型
model = Sequential()
layers = [
    Dense(12, input_dim=8, activation='relu'),
    Dense(8, activation='relu'),
    Dense(1, activation='sigmoid')
]
for layer in layers:
    model.add(layer)
# 编译模型
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# 训练模型
model.fit(X, Y, batch_size=10, epochs=150)

# 评估模型
scores = model.evaluate(X, Y)
print(scores)
