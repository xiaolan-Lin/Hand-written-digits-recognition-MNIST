from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from PIL import Image
from pylab import mpl
import matplotlib.pyplot as plt
import numpy as np
import os


def cnn_model():
    """ 载入模型 """
    model = load_model(r"D:\PycharmProjects\CNN\model\cnn_model.h5")
    return model


def num_img(filename):
    """ 模型测试 """
    img = Image.open(filename)
    img = img.convert('L')  # 灰度图
    img = np.array(img)
    img1 = img
    scaler = MinMaxScaler()
    img = scaler.fit_transform(img)
    img = np.reshape(img, (1, 28, 28, 1)).astype('float32')

    return img, img1


def test_model():
    """ 模型测试 """
    mpl.rcParams['font.sans-serif'] = ['SimHei']

    path = r"D:\PycharmProjects\CNN\after_num_image"  # 测试的图片的路径

    l = []
    after = []
    for root, dirs, files in os.walk(path):
        for name in files:
            filename = os.path.join(root, name)
            # print("图片路径")
            # print(filename)
            img, img1 = num_img(filename)
            l.append(img1)
            result = model.predict_classes(img)
            after.append(result)
            print("图片", name, "识别的数字为：", result[0])

    return l, after


def predicition(image, pred, num):
    """ 查看数字图形，标签、预测结果 """
    labels = [0, 1, 6, 8]
    fig = plt.gcf()
    fig.set_size_inches(12, 14)
    if num > 4:
        num = 4
    for i in range(0, num):
        ax = plt.subplot(2, 2, i+1)
        ax.imshow(image[i], cmap='binary')
        title = "数字为" + str(labels[i])
        if len(pred) > 0:
            title += ", 预测为" + str(pred[i][0])
        ax.set_title(title, fontsize=24)  # 设置标题
        ax.set_xticks([])  # 设置x轴和y轴的标签，这里设置不显示
        ax.set_yticks([])
    plt.show()


if __name__ == '__main__':
    model = cnn_model()
    image, pre = test_model()
    predicition(image, pre, len(image))
