from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model, model_from_json
from sklearn.preprocessing import MinMaxScaler
from keras.utils.vis_utils import model_to_dot
from IPython.display import SVG
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


"""
载入初次训练的模型，进行再训练
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

    # 给定的像素灰度值再0-255，为了使模型的训练效果更好，通常将数值归一化映射到0-1
    scaler = MinMaxScaler()  # 归一化处理
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)

    X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
    X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

    # 独热编码one-hot处理
    y_train = to_categorical(y_train)  # (60000, 10)
    y_test = to_categorical(y_test)

    return X_train, X_test, y_train, y_test


def cnn_model():
    """ 载入模型 """
    model = load_model(r"D:\PycharmProjects\CNN\model\cnn_model.h5")
    return model


def acc_model(model, X_test, y_test):
    """ 评估模型 """
    loss, accuracy = model.evaluate(X_test, y_test)

    print("loss：", loss)
    print("accuracy：", accuracy)


def train(model, X_train, y_train, X_test, y_test):
    """ 训练模型 """
    model.fit(X_train, y_train, batch_size=100, epochs=2)
    acc_model(model, X_test, y_test)


def test(model, X_test):
    """" 测试模型 """
    y_pre = model.predict_classes(X_test)
    return y_pre


def params_model(model):
    """ 网络结构 """
    # 模型保存参数
    model.save_weights('cnn_model_weight.h5')
    model.load_weights('cnn_model_weight.h5')
    # 保存网络结构
    json_string = model.to_json()
    # 载入网络结构
    model = model_from_json(json_string)
    print(json_string)
    print("model=========")

    print(model)


def crossrtab_matrix(y_test, y_pre):
    """
    交叉表、交叉矩阵
    查看预测数据与原数据对比
    """
    y_test = np.argmax(y_test, axis=1).reshape(-1)
    # print(y_test.shape)
    # print(y_pre.shape)
    crosstab = pd.crosstab(y_test, y_pre, rownames=['labels'], colnames=['predict'])
    matrix = pd.DataFrame(crosstab)
    sns.heatmap(matrix, annot=True, cmap="BuPu", fmt='d', linewidths=0.2, linecolor='pink')
    plt.show()


# def show_model(model):
#     """
#     将网络结构可视化展示出来
#     """
#     # model即为要可视化的网络模型
#     SVG(model_to_dot(model).create(prog='dot', format='svg'))


if __name__ == '__main__':
    # 载入mnist数据
    X_train, X_test, y_train, y_test = process_data()
    # 载入模型
    model = cnn_model()
    # 评估模型
    acc_model(model, X_test, y_test)
    # 训练模型
    train(model, X_train, y_train, X_test, y_test)
    # 测试模型
    y_pre = test(model, X_test)
    # 网络结构
    params_model(model)
    crossrtab_matrix(y_test, y_pre)
    # show_model(model)
