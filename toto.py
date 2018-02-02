import matplotlib
matplotlib.use('TkAgg') #truc bizarre à rajouter spécifique à mac+virtualenv
import tensorflow as tf
import ne_ne.INGREDIENTS as ing
import numpy as np
import matplotlib.pyplot as plt




def summarizeW_asImage(W):

    mat=stick_imgs(W)
    mat_shape=mat.get_shape().as_list()
    mat=tf.reshape(mat,shape=[1,mat_shape[0],mat_shape[1],1])
    print(mat.get_shape().as_list())
    tf.summary.image(W.name,mat, max_outputs=1)


def stick_imgs(W, nbChannel2=5, nbChannel3=5):

    W_shape = W.get_shape().as_list()
    nb0=min(nbChannel2, W_shape[2])
    nb1=min(nbChannel3, W_shape[3])

    sep=tf.constant(0.,shape=(1,W_shape[1]))
    columns=[]
    for j in range(nb1):
        Ws=[]
        for i in range(nb0):
            Ws.append(W[:,:,i,j])
            Ws.append(sep)
        column=tf.concat(Ws,0)
        shape_Ligne = column.get_shape().as_list()
        sep_ligne=tf.constant(0.,shape=[shape_Ligne[0],1])
        columns.append(column)
        columns.append(sep_ligne)
    mat=tf.concat(columns,1)

    return mat



init=np.ones(shape=(8,25,10,10),dtype=np.float32)
W=tf.Variable(initial_value=init)
mat=stick_imgs(W)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    mmat=sess.run(mat)
    print(mmat.shape)
    plt.imshow(mmat)
    plt.show()








