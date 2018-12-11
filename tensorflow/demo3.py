import tensorflow as tf

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#创建变量num
num = tf.Variable(0, name= "count")
#创建加法操作，把当前数字加10
new_value = tf.add(num, 10)
#创建赋值操作，把new_value赋值给num
op = tf.assign(num, new_value)

#使用这种写法，在运行完毕后，会话自动关闭
with tf.Session() as sess:

    #初始化变量
    sess.run(tf.global_variables_initializer())

    #打印最初的num的值
    print(sess.run(num))

    #创建一个for循环，每次num+10，并打印出来
    for i in range(5):
        sess.run(op)
        print(sess.run(num))

