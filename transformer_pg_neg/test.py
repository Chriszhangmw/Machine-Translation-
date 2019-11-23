import tensorflow as tf
import numpy as np

def model():
    a = tf.constant([[[2,3,4,1],
                    [3,4,7,2]],
                     [[1, 1, 2, 1],
                      [4, 0, 3, 2]]
                     ])
    c = [[[2,4,5,6],
         [4,1,7,9]],
         [[2, 4, 5, 6],
          [4, 1, 7, 9]]]
    # c = [2,4,5,6,7]
    c = tf.convert_to_tensor(c)
    c = tf.matrix_diag(c)
    print(a.get_shape())
    print(c.get_shape())
    a = tf.reshape(a,[2,2,1,4])
    # print(a.get_shape)
    k = tf.multiply(a,c)
    print(k.get_shape())
    k = tf.matrix_diag_part(k)
    print(k.get_shape())
    #
    # print(k.get_shape())
    # print(k)

    return a

with tf.Session() as sess:
    c = model()
    # print(sess.run(c))



