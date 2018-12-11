import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

v1 = tf.constant([[2,3]])
print(v1)

v2 = tf.constant([[2],[3]])
print(v2)

product = tf.matmul(v1, v2)

print(product)

#定义会话
sess = tf.Session()

# 运算成分，得到结果
result = sess.run(product)

print(result)

sess.close()
