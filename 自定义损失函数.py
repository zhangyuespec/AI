import tensorflow as tf
from numpy.random import RandomState

batch_size = 8

# 设置占位符
x = tf.placeholder(tf.float32, shape=[None, 2], name='X-input')
y = tf.placeholder(tf.float32, shape=[None, 1], name='y-input')

# 定义单层神经网络向前传播过程,初始化权值和偏置
w1 = tf.Variable(tf.random_normal([2, 1], stddev=1, seed=1))

y_predict = tf.matmul(x, w1)

# 自定义损失函数
loss_less = 10
loss_more = 1

loss = tf.reduce_sum(tf.where(tf.greater(y, y_predict), (y - y_predict) * loss_more, (y_predict - y) * loss_less))

# 优化
trian_step = tf.train.AdamOptimizer(0.0001).minimize(loss)

# 随机生成模拟数据集
rdm = RandomState(1)
data_size = 128
X = rdm.rand(data_size, 2)  # 128个样本

# 加入噪音
Y = [[x1 + x2 + rdm.rand() / 10.0 - 0.05] for (x1, x2) in X]

# 训练
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    for i in range(5000):
        start = (i * batch_size) % data_size
        end = min(start + batch_size, data_size)

        sess.run(trian_step, feed_dict={x: X[start:end], y: Y[start:end]})
        print(sess.run(w1))
