import tensorflow as tf
import numpy as np




""" Autre problème : souvent la dim-0 du tenseur d'entrée n'est pas connue car cela correspond à la taille du batch.
Cette petite fonction règle le problème.
"""
def up_convolution(X, W, stride_h, stride_w):

        nb_input_chanel=W.get_shape().as_list()[3]
        nb_output_chanel=W.get_shape().as_list()[2]

        assert X.get_shape().as_list()[3]==nb_input_chanel, "X and W are not compatible"

        input_size_h = X.get_shape().as_list()[1]
        input_size_w = X.get_shape().as_list()[2]

        output_size_h=input_size_h*stride_h
        output_size_w=input_size_w*stride_w



        """tf.shape(input)[0] c'est le batch-size, qui n'est déterminée qu'au moment du sess.run. 
          Du coup c'est un tf.shape et pas tf.get_shape """
        output_shape = tf.stack([tf.shape(X)[0],output_size_h, output_size_w,nb_output_chanel])

        upconv = tf.nn.conv2d_transpose(X, W, output_shape, [1, stride_h, stride_w, 1])

        # Now output.get_shape() is equal (?,?,?,?) which can become a problem in the
        # next layers. This can be repaired by reshaping the tensor to its shape:
        output = tf.reshape(upconv, output_shape)
        # now the shape is back to (?, H, W, C)

        return output



"""
Une bonne  manière d'initialiser le filtre de convolution-transposée : on utilise les noyaux bilinéaires
C'est les noyaux classiques que l'on utilise pour dilater une image
d'après http://cv-tricks.com/image-segmentation/transpose-convolution-in-tensorflow/"""
def get_bilinear_initial_tensor(filter_shape, upscale_factor):

    assert filter_shape[0]==filter_shape[1], "only square filters are produced here"

    kernel_size = filter_shape[1]
    """point culminant """
    if kernel_size % 2 == 1:
        centre_location = upscale_factor - 1
    else:
        centre_location = upscale_factor - 0.5

    bilinear = np.zeros([filter_shape[0], filter_shape[1]],dtype=np.float32)
    for x in range(filter_shape[0]):
        for y in range(filter_shape[1]):
            """calcul de l'interpolation"""
            value = (1 - abs((x - centre_location) / upscale_factor)) * (1 - abs((y - centre_location) / upscale_factor))
            bilinear[x, y] = value

    """on recopie pour tous les chanel"""
    weights = np.zeros(filter_shape,dtype=np.float32)
    for i in range(filter_shape[2]):
        for j in range(filter_shape[3]):
            """j'ai divisé par nb_in-chanel (dans la référence cité ci-dessus, ils n'initialisent que la diagonale, bizarre."""
            weights[:, :, i, j] = bilinear/filter_shape[2]


    return weights



""" ici :https://stackoverflow.com/questions/33712178/tensorflow-nan-bug
ils indique que faire un clip sur Y_hay n'est pas une bonne idée car cela arrète la propagation du gradient... 
J'ai tester  :  un clip  "grossier"  ex : tf.clip_by_value(Y_hat, 0.0001, 0.9999)  tue vraiment l'apprentissage.
Un clip plus fin ne change pas grand chose.
   """
def crossEntropy(Y, Y_hat, avoidNan=True):
    if avoidNan:  Y_hat+=1e-10
    return - tf.reduce_mean(Y*tf.log(Y_hat))


def quadraticLoss(Y, Y_hat):
    return tf.reduce_mean((Y-Y_hat)**2)



"""
Attention ici au fameux  1+1e-10 -Y avec Y=1  ---> 0  et quand on passe au log ---> nan
"""
def crossEntropy_multiLabel(Y,Y_hat):
    return - tf.reduce_mean(  Y * tf.log(Y_hat+1e-10) + (1 - Y) * tf.log( (1 - Y_hat)+1e-10))



def accuracy(true_Y_proba, hat_Y_proba):
    return tf.reduce_mean(tf.cast(tf.equal(hat_Y_proba, true_Y_proba), tf.float32))

#
# def make_binary_with_arg_max(Y_hat,nb_category):
#     return tf.one_hot(tf.arg_max(Y_hat, dimension=1), nb_category )
#
# def make_binary_with_threshold(Y_hat,threshold):
#     return tf.cast(tf.greater(Y_hat, threshold), tf.float32)



def conv2d_basic(x, W, bias):
    conv = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME")
    return tf.nn.bias_add(conv, bias)


def conv2d_strided(x, W, b):
    conv = tf.nn.conv2d(x, W, strides=[1, 2, 2, 1], padding="SAME")
    return tf.nn.bias_add(conv, b)



def conv2d_transpose_strided(x, W, b, output_shape=None, stride = 2):
    # print x.get_shape()
    # print W.get_shape()
    if output_shape is None:
        output_shape = x.get_shape().as_list()
        output_shape[1] *= 2
        output_shape[2] *= 2
        output_shape[3] = W.get_shape().as_list()[2]
    # print output_shape
    conv = tf.nn.conv2d_transpose(x, W, output_shape, strides=[1, stride, stride, 1], padding="SAME")
    return tf.nn.bias_add(conv, b)




""" attention, stddev=0.1 est très important. 
Par exemple, sur le modèle bounding box avec les données squareCircle, cela ne converge pas avec stddev=0.02.
Une initialisation standard est stddev=sqrt(2/nbInput). Mais avec nbInput =1024 comme dans leNet, cela fait 0.05: très petit. 
"""
def weight_variable(shape, stddev=0.1, name=None):
    initial = tf.truncated_normal(shape, stddev=stddev)
    return tf.Variable(initial_value=initial,name=name)



"""Dans le cours de Stanfort, Karpathy indique que cela serait mieux en prenant des biais nul. """
def bias_variable(shape, name=None):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial_value=initial,name=name)




def leaky_relu(x, alpha=0.0, name=""):
    return tf.maximum(alpha * x, x, name)


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


def avg_pool_2x2(x):
    return tf.nn.avg_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


def local_response_norm(x):
    return tf.nn.lrn(x, depth_radius=5, bias=2, alpha=1e-4, beta=0.75)


def batch_norm(x, n_out, phase_train, scope='bn', decay=0.9, eps=1e-5):
    """
    Code taken from http://stackoverflow.com/a/34634291/2267819
    """
    with tf.variable_scope(scope):
        beta = tf.get_variable(name='beta', shape=[n_out], initializer=tf.constant_initializer(0.0)
                               , trainable=True)
        gamma = tf.get_variable(name='gamma', shape=[n_out], initializer=tf.random_normal_initializer(1.0, 0.02),
                                trainable=True)
        batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=decay)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(phase_train,
                            mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, eps)
    return normed



def summarizeW(W,name="W",nbImg=4):
    if nbImg>0 :
        W_tr=tf.transpose(W, [2, 0, 1, 3])
        tf.summary.image(name,W_tr[:,:,:,:1] , max_outputs=nbImg)
    tf.summary.histogram(name, W)
    tf.summary.scalar(name, tf.nn.zero_fraction(W))


#
# """d'après http://cv-tricks.com/image-segmentation/transpose-convolution-in-tensorflow/"""
# def get_bilinear_filter(filter_shape, upscale_factor):
#     ##filter_shape is [width, height, num_in_channels, num_out_channels]
#     kernel_size = filter_shape[1]
#     ### Centre location of the filter for which value is calculated
#     if kernel_size % 2 == 1:
#         centre_location = upscale_factor - 1
#     else:
#         centre_location = upscale_factor - 0.5
#
#     bilinear = np.zeros([filter_shape[0], filter_shape[1]])
#     for x in range(filter_shape[0]):
#         for y in range(filter_shape[1]):
#             ##Interpolation Calculation
#             value = (1 - abs((x - centre_location) / upscale_factor)) * (1 - abs((y - centre_location) / upscale_factor))
#             bilinear[x, y] = value
#     weights = np.zeros(filter_shape)
#     for i in range(filter_shape[2]):
#         weights[:, :, i, i] = bilinear
#     init = tf.constant_initializer(value=weights,dtype=tf.float32)
#
#     bilinear_weights = tf.get_variable(name="decon_bilinear_filter", initializer=init,
#                                        shape=weights.shape)
#     return bilinear_weights
#
#
#
# def upsample_layer(bottom, n_channels, name, upscale_factor):
#
#     kernel_size = 2 * upscale_factor - upscale_factor % 2
#     stride = upscale_factor
#     strides = [1, stride, stride, 1]
#     with tf.variable_scope(name):
#         # Shape of the bottom tensor
#         in_shape = tf.shape(bottom)
#
#         h = ((in_shape[1] - 1) * stride) + 1
#         w = ((in_shape[2] - 1) * stride) + 1
#         new_shape = [in_shape[0], h, w, n_channels]
#         output_shape = tf.stack(new_shape)
#
#         filter_shape = [kernel_size, kernel_size, n_channels, n_channels]
#
#         weights = get_bilinear_filter(filter_shape, upscale_factor)
#         deconv = tf.nn.conv2d_transpose(bottom, weights, output_shape,strides=strides, padding='SAME')
#
#     return deconv
#
#
#
#
# def upconvolution(input, output_channel_size, filter_size_h, filter_size_w,
#                   stride_h, stride_w, init_w, init_b, layer_name,
#                   dtype=tf.float32, data_format="NHWC", padding='VALID'):
#     with tf.variable_scope(layer_name):
#         # calculation of the output_shape:
#         if data_format == "NHWC":
#             input_channel_size = input.get_shape().as_list()[3]
#             input_size_h = input.get_shape().as_list()[1]
#             input_size_w = input.get_shape().as_list()[2]
#             stride_shape = [1, stride_h, stride_w, 1]
#             if padding == 'VALID':
#                 output_size_h = (input_size_h - 1) * stride_h + filter_size_h
#                 output_size_w = (input_size_w - 1) * stride_w + filter_size_w
#             elif padding == 'SAME':
#                 output_size_h = (input_size_h - 1) * stride_h + 1
#                 output_size_w = (input_size_w - 1) * stride_w + 1
#             else:
#                 raise ValueError("unknown padding")
#             output_shape = tf.stack([tf.shape(input)[0],
#                                      output_size_h, output_size_w,
#                                      output_channel_size])
#         elif data_format == "NCHW":
#             input_channel_size = input.get_shape().as_list()[1]
#             input_size_h = input.get_shape().as_list()[2]
#             input_size_w = input.get_shape().as_list()[3]
#             stride_shape = [1, 1, stride_h, stride_w]
#             if padding == 'VALID':
#                 output_size_h = (input_size_h - 1) * stride_h + filter_size_h
#                 output_size_w = (input_size_w - 1) * stride_w + filter_size_w
#             elif padding == 'SAME':
#                 output_size_h = (input_size_h - 1) * stride_h + 1
#                 output_size_w = (input_size_w - 1) * stride_w + 1
#             else:
#                 raise ValueError("unknown padding")
#             output_shape = tf.stack([tf.shape(input)[0],
#                                      output_channel_size,
#                                      output_size_h, output_size_w])
#         else:
#             raise ValueError("unknown data_format")
#
#         # creating weights:
#         shape = [filter_size_h, filter_size_w,
#                  output_channel_size, input_channel_size]
#         W_upconv = tf.get_variable("w", shape=shape, dtype=dtype,
#                                    initializer=init_w)
#
#         shape = [output_channel_size]
#         b_upconv = tf.get_variable("b", shape=shape, dtype=dtype,
#                                    initializer=init_b)
#
#         upconv = tf.nn.conv2d_transpose(input, W_upconv, output_shape, stride_shape,
#                                         padding=padding,
#                                         data_format=data_format)
#         output = tf.nn.bias_add(upconv, b_upconv, data_format=data_format)
#
#         # Now output.get_shape() is equal (?,?,?,?) which can become a problem in the
#         # next layers. This can be repaired by reshaping the tensor to its shape:
#         output = tf.reshape(output, output_shape)
#         # now the shape is back to (?, H, W, C) or (?, C, H, W)
#
#         return output
#
#

















#
# def _upscore_layer(self, bottom, shape,
#                    num_classes, name, debug,
#                    ksize=4, stride=2):
#     strides = [1, stride, stride, 1]
#
#     with tf.variable_scope(name):
#         in_features = bottom.get_shape()[3].value
#
#         if shape is None:
#             # Compute shape out of Bottom
#             in_shape = tf.shape(bottom)
#
#             h = ((in_shape[1] - 1) * stride) + 1
#             w = ((in_shape[2] - 1) * stride) + 1
#             new_shape = [in_shape[0], h, w, num_classes]
#         else:
#             new_shape = [shape[0], shape[1], shape[2], num_classes]
#         output_shape = tf.stack(new_shape)
#
#         # logging.debug("Layer: %s, Fan-in: %d" % (name, in_features))
#         f_shape = [ksize, ksize, num_classes, in_features]
#
#         # create
#         num_input = ksize * ksize * in_features / stride
#         stddev = (2 / num_input)**0.5
#
#         weights = self.get_deconv_filter(f_shape)
#         deconv = tf.nn.conv2d_transpose(bottom, weights, output_shape,
#                                         strides=strides, padding='SAME')
#
#         if debug:
#             deconv = tf.Print(deconv, [tf.shape(deconv)],
#                               message='Shape of %s' % name,
#                               summarize=4, first_n=1)
#
#     return deconv
#
#
#
# def get_deconv_filter(f_shape):
#     width = f_shape[0]
#     height = f_shape[1]
#     f = np.ceil(width/2.0)
#     c = (2 * f - 1 - f % 2) / (2.0 * f)
#     bilinear = np.zeros([f_shape[0], f_shape[1]])
#     for x in range(width):
#         for y in range(height):
#             value = (1 - abs(x / f - c)) * (1 - abs(y / f - c))
#             bilinear[x, y] = value
#     weights = np.zeros(f_shape)
#     for i in range(f_shape[2]):
#         weights[:, :, i, i] = bilinear
#
#     init = tf.constant_initializer(value=weights,
#                                    dtype=tf.float32)
#     return tf.get_variable(name="up_filter", initializer=init,
#                            shape=weights.shape)
#




