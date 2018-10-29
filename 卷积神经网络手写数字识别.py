import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

def weight_variable(shape):
    """
    初始化权重的函数
    :shape:形状
    :return: w
    """
    w=tf.Variable(tf.random_normal(shape=shape,mean=0.0,stddev=1.0))
    return w


def bias_varible(shape):
    """
    定义初始化偏置函数
    :param shape: 形状
    :return: b
    """
    bias=tf.Variable(tf.constant(0.0,shape=shape))
    return bias

def model():
    """
    自定义卷积模型
    :return:
    """
    #准备占位符
    with tf.variable_scope("data"):
        x=tf.placeholder(tf.float32,[None,784])
        y_true=tf.placeholder(tf.int32,[None,10])

    #卷积层1,卷积激活池化
    with tf.variable_scope("con1"):
        #对x形状改变
        x_reshape=tf.reshape(x,[-1,28,28,1])

        shape_weight=[5,5,1,32]
        shape_bias=[32]
        weight=weight_variable(shape_weight)

        biases=bias_varible(shape_bias)

        #卷积操作
        con=tf.nn.conv2d(x_reshape,weight,strides=[1,1,1,1],padding="SAME")

        #加上偏置
        con=con+biases

        #激活
        actived_conv=tf.nn.relu(con)

        #池化

        pool = tf.nn.max_pool(actived_conv, ksize=[1, 2,2, 1], strides=[1, 2, 2, 1], padding="SAME")

    #卷积层2
    with tf.variable_scope("con2"):
        shape_weight=[5,5,32,64]
        biases=bias_varible([64])
        weight=weight_variable(shape_weight)

        #卷积，激活，池化
        pool=tf.nn.max_pool(tf.nn.relu(tf.nn.conv2d(pool,weight,strides=[1,1,1,1],padding="SAME")+biases),
                            ksize=[1, 2,2, 1], strides=[1, 2, 2, 1], padding="SAME")


    #全连接层
    with tf.variable_scope("full_con"):
        w_fc=weight_variable([7*7*64,10])
        b_fc=bias_varible([10])


        x_fc_reshape=tf.reshape(pool,[-1,7*7*64])
        #进行矩阵运算，得出每个样本计算结果
        y_predict=tf.matmul(x_fc_reshape,w_fc)+b_fc

    return x,y_true,y_predict



def conv2():
    """
    卷积实现手写数字识别
    :return: None
    """
    mnist = input_data.read_data_sets("./data", one_hot=True)
    #定义模型得出输出,第一步的向前传播
    x, y_true, y_predict=model()


    with tf.variable_scope("soft_max"):
        # 求平均交叉熵损失
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_predict))

    # 梯度下降求出损失
    with tf.variable_scope("optimaizer"):
        train_op = tf.train.GradientDescentOptimizer(0.001).minimize(loss)

    # 计算准确率
    with tf.variable_scope("acc"):
        equal_list = tf.equal(tf.argmax(y_true, 1), tf.argmax(y_predict, 1))

        accuracy = tf.reduce_mean(tf.cast(equal_list, tf.float32))

    #开启绘画运行
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for i in range(1000):
            # 取出真实存在的特征值和目标值
            mnist_x, mnist_y = mnist.train.next_batch(50)

            # 运行训练
            sess.run(train_op, feed_dict={x: mnist_x, y_true: mnist_y})

            # 写入每步训练的值
            # summary = sess.run(merged, feed_dict={x: mnist_x, y_true: mnist_y})
            # filewrite.add_summary(summary, i)

            print("训练第%d步，准确率为%f" % (i, sess.run(accuracy, feed_dict={x: mnist_x, y_true: mnist_y})))

if __name__ == '__main__':
    conv2()