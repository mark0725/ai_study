import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 创建两个变量占位符
input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)

# 创建一个乘法操作，把input1和input2相乘
new_value = tf.multiply(input1, input2)

#使用这种写法，在运行完毕后，会话自动关闭
with tf.Session() as sess:

    # 打印new_value,在运算时，用feed设置两个输入的值
    print(sess.run(new_value, feed_dict={input1:23.0, input2:11.0}))

   