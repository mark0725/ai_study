import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

v1 = tf.constant([[2,3]])
v2 = tf.constant([[2],[3]])
product = tf.matmul(v1, v2)

#定义会话
with tf.Session() as sess:
    writer = tf.summary.FileWriter("logs/", sess.graph)
    
    # 运算成分，得到结果
    result = sess.run(product)
    print(result)
