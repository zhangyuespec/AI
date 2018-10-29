import tensorflow as tf

# 初始化权重矩阵，尺寸5*5，当前层深度是3，过滤器深度是16就代表有16个filter

filter_weight = tf.get_variable("weight", [5, 5, 3, 16], initializer=tf.truncated_normal_initializer(stddev=0.1))

# 为16个过滤器设置偏置
biases = tf.get_variable("biases", [16], initializer=tf.constant(0.1))

# tf.nn.connv2d提供了非常方便的函数实现了卷积层的向前传播。，这个函数第一个矩阵代表当前层节点矩阵，这个矩阵是一个四维矩阵
# 后面三个维度对应节点矩阵，第一个维度对用哪一张图片，(batch批处理)
conv = tf.nn.conv2d(input, filter_weight, strides=[1, 1, 1, 1], padding="SAME")  # SAME代表选择0填充

# 加上偏置
bias = tf.nn.bias_add(conv, biases)

# ReLU激活函数去线性化

actived_conv = tf.nn.relu(bias)

#池化层
pool=tf.nn.max_pool(actived_conv,ksize=[1,3,3,1],strides=[1,2,2,1],padding="SAME")