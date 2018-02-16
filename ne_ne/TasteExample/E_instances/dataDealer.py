import numpy as np
import matplotlib

from ne_ne.dataGen.dataLines.weighted_line import weighted_line
matplotlib.use('TkAgg') #truc bizarre à rajouter spécifique à mac+virtualenv
import matplotlib.pyplot as plt

#
# def norm1(a):
#     return np.sum(np.sum(a,axis=0),axis=0)


#
# def addRectangle(size,position,img,cat):
#     b1=min(position[1]+size[1],img.shape[1])
#     b0=min(position[0]+size[0],img.shape[0])
#
#     img[position[1]:b1,position[0]:b0]=cat



#
#
#
# def make_img(img_size,nb_cat):
#
#     img = np.zeros([img_size, img_size],dtype=np.int32)
#
#     for i in range(1, nb_cat + 1):
#         rect_size = [np.random.randint(0, 10), np.random.randint(0, 10)]
#         position = [np.random.randint(0, img_size - rect_size[0]), np.random.randint(0, img_size - rect_size[0])]
#         addRectangle(rect_size, position, img, i)
#     return img






#
#
# def make_instances_GT(img_size:int, nb_instances_max:int, withBackground:bool):
#
#     img = make_img(img_size, nb_instances_max)
#
#     eye = np.eye(nb_instances_max + 1, dtype=np.float32)
#     Y = eye[img]
#
#     Y_norm = norm1(Y)
#     non_zero = Y_norm > 0
#
#     Y_mod = Y[:, :, non_zero]
#
#     if not withBackground:
#         Y_mod=Y_mod[:,:,1:]
#
#     return Y_mod,img


def addOneLine(point0:tuple, point1:tuple, img:np.ndarray, Y_class:np.ndarray,Y_reg:np.ndarray,cat:int):
    yy, xx, vals = weighted_line(point0[0], point0[1], point1[0], point1[1], 0)

    dx = point1[0] - point0[0]
    dy = point1[1] - point0[1]

    angle = np.arctan2(dx, dy)

    Y_class[xx, yy] = cat

    Y_reg[xx, yy] =angle % np.pi

    """ les images sont initialisées à 1."""
    img[xx, yy] *= 1 - vals


def batch_of_lines_Y_is_instanceNumber(img_size:int, batchSize:int,nbLinesPerImg):

    imgs = np.ones([batchSize, img_size, img_size], dtype=np.float32)
    Ys_class = np.zeros([batchSize, img_size, img_size], dtype=np.uint8)
    Ys_reg = np.zeros([batchSize, img_size, img_size], dtype=np.float32)
    Ys_reg[:]=np.nan

    for b in range(batchSize):

        for j in range(nbLinesPerImg):
            KO=True
            while KO:
                point0 = (np.random.randint(1, img_size - 1), np.random.randint(1, img_size - 1))
                point1 = (np.random.randint(1, img_size - 1), np.random.randint(1, img_size - 1))
                KO= (point0==point1)
                """ +1 car 0 c'est le background  """
                if not KO : addOneLine(point0=point0, point1=point1, img=imgs[b], Y_class=Ys_class[b], Y_reg=Ys_reg[b] , cat=j+1)

    return np.expand_dims(imgs,3),Ys_class,Ys_reg




def oneBatch_lines_instances(batchSize, img_size, nbLinesPerImage, withBackground):

    Xs,Ys,_=batch_of_lines_Y_is_instanceNumber(batchSize=batchSize,img_size=img_size,nbLinesPerImg=nbLinesPerImage)

    eye = np.eye(nbLinesPerImage + 1, dtype=np.float32)
    Ys = eye[Ys]

    if not withBackground:
        Ys=Ys[:,:,:,1:]

    return Xs,Ys

#
#
# def make_instances_GT_batch(img_size, nb_cat_max, batch_size, withBackground:bool,nbAdditionnalStrate:int):
#
#     X=np.zeros([batch_size,img_size,img_size],dtype=np.float32)
#
#     nbCat=nb_cat_max+nbAdditionnalStrate
#
#     if withBackground : nbCat+=1
#     Y=np.zeros([batch_size,img_size,img_size,nbCat],dtype=np.float32)
#
#     for i in range(batch_size):
#         Yi,Xi=make_instances_GT(img_size, nb_cat_max, withBackground)
#
#         X[i,:,:]=Xi
#         nbCat_present=Yi.shape[2]
#         Y[i,:,:,:nbCat_present]=Yi
#
#     return np.expand_dims(X,3),Y




def test():
    img_size = 28
    nbInstances=3
    withBackground=False


    Xs,Ys=oneBatch_lines_instances(1, img_size, nbInstances, withBackground)
    X,Y=Xs[0],Ys[0]
    print("X.shape,Y.shape",X.shape,Y.shape)
    nb_cat = Y.shape[2]


    plt.figure()
    plt.imshow(np.reshape(X,[img_size,img_size]),cmap="gray")
    plt.colorbar()

    plt.figure()
    """ attention, argmax mange la classe 0 ! """
    plt.imshow(np.argmax(Y,axis=2), cmap="jet")
    plt.colorbar()


    plt.figure()

    for i in range(nb_cat):
        plt.subplot(4,4,i+1)
        plt.imshow(Y[:,:,i],cmap="gray")
        plt.title("cat:"+str(i))

    plt.show()


if __name__=="__main__":
    test()
