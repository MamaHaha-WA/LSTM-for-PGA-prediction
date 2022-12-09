# -*- coding: utf-8 -*-
# @Author  : WA
# @FileName: getLSTM.py
# @Software: PyCharm
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense, Masking


def get_LSTM(time_step, features, layers=4, units=4, optimizer='nadam', a='relu', a_l='tanh', loss='mse'):
    """
    搭建一个LSTM循环神经网络
    @param time_step: 时间步
    @param features: 输入的特征数目
    @param layers: 神经网络的深度(LSTM层的层数)
    @param units: 首个卷积层的宽度，即卷积神经元的数目
    @param optimizer: 优化器，如Adam，Ndam
    @param a: 激活函数，如ReLU，sigmoid
    @param a_l: LSTM层激活函数，如tanh, softsign
    @param loss: 损失函数，如mse，msle
    @return: model
    """
    model = Sequential()
    model.add(Masking(mask_value=0., input_shape=(time_step, features)))
    if layers == 5:
        model.add(LSTM(16 * units, activation=a_l, return_sequences=True))
        model.add(LSTM(8 * units, activation=a_l, return_sequences=True))
        model.add(LSTM(4 * units, activation=a_l, return_sequences=True))
    # model.add(Dropout(0.1))
    elif layers == 4:
        model.add(LSTM(8 * units, activation=a_l, return_sequences=True))
        model.add(LSTM(4 * units, activation=a_l, return_sequences=True))
    # model.add(Dropout(0.1))
    elif layers == 3:
        model.add(LSTM(4 * units, activation=a_l, return_sequences=True))
    # model.add(Dropout(0.1))
    model.add(LSTM(2 * units, activation=a_l, return_sequences=True))
    # model.add(Dropout(0.1))
    model.add(LSTM(units, activation=a_l, return_sequences=False))
    model.add(Dense(1, activation=a))

    model.compile(loss=loss, optimizer=optimizer)
    return model


if __name__ == '__main__':
    Model = get_LSTM(100, 8, 4, 4)
    Model.summary()
