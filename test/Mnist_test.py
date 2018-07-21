
import numpy as np
import struct
 
def loadImageSet(filename):
    binfile = open(filename, 'rb') # 读取二进制文件
    buffers = binfile.read()
    head = struct.unpack_from('>IIII', buffers, 0) # 取前4个整数，返回一个元组
    offset = struct.calcsize('>IIII')  # 定位到data开始的位置
    imgNum = head[1]
    width = head[2]
    height = head[3]
    bits = imgNum * width * height  # data一共有60000*28*28个像素值
    bitsString = '>' + str(bits) + 'B'  # fmt格式：'>47040000B'
    imgs = struct.unpack_from(bitsString, buffers, offset) # 取data数据，返回一个元组
    binfile.close()
    imgs = np.reshape(imgs, [imgNum, width * height]) # reshape为[60000,784]型数组
    return imgs,head
 
def loadLabelSet(filename):
    binfile = open(filename, 'rb') # 读二进制文件
    buffers = binfile.read()
    head = struct.unpack_from('>II', buffers, 0) # 取label文件前2个整形数
    labelNum = head[1]
    offset = struct.calcsize('>II')  # 定位到label数据开始的位置
    numString = '>' + str(labelNum) + "B" # fmt格式：'>60000B'
    labels = struct.unpack_from(numString, buffers, offset) # 取label数据
    binfile.close()
    labels = np.reshape(labels, [labelNum]) # 转型为列表(一维数组)
    return labels,head
 
if __name__ == "__main__":
    file1= 'E:/Python_workspace/mnist-master/Mnist_data/train-images-idx3-ubyte.gz'
    file2= 'E:/Python_workspace/mnist-master/Mnist_data/train-labels-idx1-ubyte.gz'
    imgs,data_head = loadImageSet(file1)
    print('data_head:',data_head)
    print(type(imgs))
    print('imgs_array:',imgs)
    print(np.reshape(imgs[1,:],[28,28])) #取出其中一张图片的像素，转型为28*28，大致就能从图像上看出是几啦
    print('----------我是分割线-----------')
    labels,labels_head = loadLabelSet(file2)
    print('labels_head:',labels_head)
    print(type(labels))

# from tensorflow.examples.tutorials.mnist import input_data
# from PIL import Image
# import numpy as np
# import os
# import matplotlib.pyplot as plt
# 
# data_sets = input_data.read_data_sets('E:/Python_workspace/mnist-master/Mnist_data/')
# images = data_sets.train.images
# labels = data_sets.train.labels
# total = images.shape[0]
# print(images.shape)
# print(images)
# im = images[7]
# im2 = np.array(im)
# print(im)
# im2 = im2.reshape(28,28)
# print(im2)
# fig = plt.figure()
# plotwindow = fig.add_subplot(1,1,1)
# plt.imshow(im2, cmap='gray')
# plt.show()

# save_path = 'E:/Python_workspace/mnist-master/pic/'
# for i in range(0, total):
#     label = labels[i]
#     path = os.path.join(save_path, str(label))
#     if not os.path.exists(path):
#         os.makedirs(path)
#     name = str(i)+'.bmp'
#     filename = os.path.join(path, name)
#     if os.path.exists(filename):
#         print(filename, 'exists')
#         continue
#     image = images[i]
#     image = image.reshape(28, 28)
#     image = numpy.multiply(image, 255)
#     image = image.astype(numpy.uint8)
#     im=Image.fromarray(image)
#     im.save(filename)
#     print(filename, 'saved')







