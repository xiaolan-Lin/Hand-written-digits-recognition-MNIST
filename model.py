from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from tensorflow.keras import backend as tkb
import tensorflow as tf


"""
模型设计成功后将模型保存为h5文件，以便于后续调用
"""


def create_dataset():
    """
    导入数据集
    """
    (X_train, y_train), (X_test, y_test) = mnist.load_data()  # 加载数据，加载出来的数据是tuple
    return X_train, X_test, y_train, y_test


def process_data():
    """
    图片数据预处理
    """
    X_train, X_test, y_train, y_test = create_dataset()  # 导入数据
    # 数据集是3维的向量
    num_pix = X_train.shape[1] * X_train.shape[2]

    X_train = X_train.reshape(X_train.shape[0], num_pix).astype('float32')
    X_test = X_test.reshape(X_test.shape[0], num_pix).astype('float32')

    print("训练集：")
    print(X_train)
    print("训练集维度：")
    print(X_train.shape)

    print("测试集：")
    print(X_test)
    print("测试集维度：")
    print(X_test.shape)

    # 给定的像素灰度值再0-255，为了使模型的训练效果更好，通常将数值归一化映射到0-1
    scaler = MinMaxScaler()  # 归一化处理
    X_train = scaler.fit_transform(X_train)
    # print("MinMaxScaler_trans_X_train:")
    # print(X_train)
    X_test = scaler.fit_transform(X_test)
    # print("MinMaxScaler_trans_X_test:")
    # print(X_test)

    X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)  # 28*28图像大小，1为通道数，60000表示图像的张数
    X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

    # print("归一化处理后的训练集：")
    # print(X_train)
    # print("归一化处理后的训练集维度：")
    # print(X_train.shape)
    # print("归一化处理后的测试集：")
    # print(X_test)
    # print("归一化处理后的测试集维度：")
    # print(X_test.shape)
    # 独热编码one-hot处理
    y_train = to_categorical(y_train)  # (60000, 10)
    # print("经过one-hot处理后的训练集标签：")
    # print(y_train)
    y_test = to_categorical(y_test)
    # print("经过one-hot处理后的训练集标签：")
    # print(y_test)

    return X_train, X_test, y_train, y_test


def cnn_model(X_train, y_test):
    """
    构建模型
    """
    num_classes = y_test.shape[1]  # 10
    # print("y_test的维度为：", y_test.shape)
    print("num_classes为：", num_classes)
    # input_shape = X_train.shape[1:]
    # print("INPUT输入层input_shape：", X_train.shape)
    # print("输入模型的图片数据维度为：", input_shape)
    # model = Sequential()
    # tkb.set_image_data_format("channels_first")  # 设置图像数据格式约定的值，tensorflow/python/keras/backend.py
    # tkb.set_image_data_format("channels_last")
    # INPUT输入层
    input_shape = X_train.shape[1:]
    # print("INPUT输入层input_shape：", X_train.shape)
    # print("输入模型的图片数据维度为：", input_shape)
    # # C1-卷积层，卷积核种类？6个滤波器，即filters
    # model.add(Conv2D(filters=6, kernel_size=ks, input_shape=input_shape, activation='relu'))  # 28*28 → 24*24
    # # S2池化层
    # model.add(MaxPooling2D(pool_size=(2, 2)))  # 24*24 → 12*12
    # # C3卷积层
    # model.add(Conv2D(filters=16, kernel_size=ks, activation="relu"))  # 12*12 → 8*8
    # # S4池化层
    # model.add(MaxPooling2D(pool_size=(2, 2)))  # 8*8 → 4*4
    # # C5卷积层
    # model.add(Conv2D(filters=120, kernel_size=ks1, activation='relu'))  # 4*4 →
    # # 对参数进行正则化防止模型过拟合
    # model.add(Dropout(0.2))
    # # 压平后可进行全连接
    # model.add(Flatten())
    # # F6全连接层
    # model.add(Dense(84, activation='relu'))
    # # Output层全连接层
    # model.add(Dense(num_classes, activation='softmax'))

    model = Sequential()
    # 两层3x3代替一层
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(32, kernel_size=(5, 5), strides=2, padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, kernel_size=(5, 5), strides=2, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv2D(128, kernel_size=(4, 4), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dropout(0.4))
    model.add(Dense(10, activation='softmax'))

    print(model.summary())

    return model


def show_train_history(train_history, train, validation):
    """ 可视化查看参数预测 """
    plt.xlim((0, 10))
    plt.ylim((0, 1))
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title("Train History")
    plt.ylabel("train")
    plt.xlabel("epoch")
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()


def train_model(model, X_train, X_test, y_train, y_test):
    """ 训练模型 """
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    train_history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=256, verbose=2)

    show_train_history(train_history, 'accuracy', 'val_accuracy')
    show_train_history(train_history, 'loss', 'val_loss')

    return model


def score_model(model, X_test, y_test):
    """ 模型评估 """
    loss, accurary = model.evaluate(X_test, y_test, verbose=0)
    print("\ntest loss:", loss)
    print("accurary:", accurary)


def save_model(model):
    """ 模型可视化 """
    plot_model(model, to_file=r"D:\PycharmProjects\CNN\model\cnn_model.png")


if __name__ == '__main__':
    # 载入数据
    X_train, X_test, y_train, y_test = process_data()
    # 构建模型
    cnn = cnn_model(X_train, y_test)
    # 训练模型
    cnn_model = train_model(cnn, X_train, X_test, y_train, y_test)
    # 保存模型
    cnn_model.save(r"D:\PycharmProjects\CNN\model\cnn_model.h5")  # HDF5文件，pip install h5py
    # 评估模型
    score_model(cnn_model, X_test, y_test)
    # 保存模型图
    save_model(cnn_model)







# （1）模型结构（参考LeNet-5）
# INPUT层-输入层
# 数据输入层，输入图像的尺寸统一化归一化为28*28。
# C1层-卷积层
# 	输入图片：28*28
# 	卷积核大小：3x3
# 卷积核种类：32
# 输出featuremap大小：26x26
# 神经元数量：26x26x6
# 可训练参数：（3x3+1）x32（每个滤波器5x5=9个unit参数和一个--bais参数，一共32个滤波器）
# 连接数：（3x3+1）x32x26x26=4216320
# C2层-卷积层
# 	输入图片：26*26
# 	卷积核大小：3x3
# 卷积核种类：32
# 输出featuremap大小：24x24
# 神经元数量：24x24x32
# 可训练参数：（3x3+1）x32（每个滤波器3x3=9个unit参数和一个--bais参数，一共32个滤波器）
# 连接数：（3x3+1）x32x24x24=184320
# C3层-卷积层
# 	输入图片：24*24
# 	卷积核大小：5x5
# 卷积核种类：32
# 输出featuremap大小：10x10（步长为2）（24-5）/2+1
# 神经元数量：10x10x32
# 可训练参数：（5x5+2）x32（每个滤波器5x5=25个unit参数和一个--bais参数，一共32个滤波器）
# 连接数：（5x5+2）x32x10x10=86400
# C4层-卷积层
# 输入图片：10*10
# 卷积核大小：3x3
# 卷积核种类：64
# 输出featureMap大小：8x8
# 神经元数量：8x8x64
# 可训练参数：（3x3+1）x64（每个滤波器3x3=9个unit参数和一个--bais参数，一共64个滤波器）
# 连接数：（3x3+1）x64x8x8=40960
# C5层-卷积层
# 输入：8x8
# 卷积核大小：3x3
# 卷积核种类：64
# 输出featureMap大小：6x6
# 神经元数量：6x6x64
# 可训练参数：（3x3+1）x64（每个滤波器3x3=9个unit参数和一个--bais参数，一共64个滤波器）
# 连接数：（3x3+1）x64x6x6=23040
# C6层-卷积层
# 输入：6x6
# 卷积核大小：5x5
# 卷积核种类：64
# 输出featureMap大小：6x6（6-5/2+1）
# 神经元数量：6x6x64
# 可训练参数：（3x3+1）x64（每个滤波器3x3=9个unit参数和一个--bais参数，一共64个滤波器）
# 连接数：（3x3+1）x64x6x6=23040
