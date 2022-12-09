# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 14:52:57 2021

@author: dell
"""
from set_seeds import set_global_determinism
import tensorflow as tf
from tensorflow.keras import callbacks
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import BatchNormalization
from pathlib import Path
from getLSTM import get_LSTM

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False

SEED = 666
# Call the above function with seed value
set_global_determinism(seed=SEED)

# 设置数据集的路径
path_data = 'F:/LSTM_Onsite_algorithm/Dataset/'
train_x = 'trainx8_1.dat'
train_y = 'trainy_PGA_18.dat'
valid_x = 'validx8_1.dat'
valid_y = 'validy_PGA_18.dat'
test_x = 'testx8_1.dat'
test_y = 'testy_PGA_18.dat'
path_example = 'F:/LSTM_Onsite_algorithm/震例/'
example_x = 'x8_1.dat'

# 设置超参数
TIME_STEP = 100
FEATURES = 8
layers = 4  # LSTM的层数
units = 4  # 底层LSTM的神经元数目
loss = 'mse'  # 损失函数
opt = 'nadam'  # 优化器
a = 'relu'  # dense层的激活函数
a_l = 'tanh'  # lstm层的激活函数
batch_size = 512  # 批处理大小
epoch = 1000  # 最大训练轮次
model_name = 'lstm_f8_lay4_PGA_6'  # 保存模型的名字
save_name = 'f8l4_pred_6'
log_num = '68'

# ------------------------------------------------------------------------------
model = get_LSTM(TIME_STEP, FEATURES, layers=layers, units=units, optimizer=opt, a=a, a_l=a_l, loss=loss)
# ------------------------------------------------------------------------------
# 读取训练集和验证集数据
X_train = np.loadtxt(path_data + train_x)
Y_train = np.loadtxt(path_data + train_y)
X_train = np.reshape(X_train, [np.size(X_train, 0), TIME_STEP, FEATURES])
Y_train = Y_train[:, 0]

X_valid = np.loadtxt(path_data + valid_x)
Y_valid = np.loadtxt(path_data + valid_y)
X_valid = np.reshape(X_valid, [np.size(X_valid, 0), TIME_STEP, FEATURES])
Y_valid = Y_valid[:, 0]
# ------------------------------------------------------------------------------
# model check point
callbacks = [
    callbacks.ModelCheckpoint('Models/'+model_name+'.h5', monitor='val_loss', verbose=1, save_best_only=True),
    callbacks.EarlyStopping(patience=15, monitor='val_loss'),
    callbacks.TensorBoard(log_dir='logs/'+log_num)]
results = model.fit(X_train, Y_train, validation_data=(X_valid, Y_valid), batch_size=batch_size, epochs=epoch,
                    callbacks=callbacks)

# ------------------------------------------------------------------------------
# 绘制训练过程的损失历史
plt.figure()
lenth = len(results.history['loss'])
plt.plot(range(1, lenth + 1), results.history['loss'], label='train')
plt.plot(range(1, lenth + 1), results.history['val_loss'], label='valid')
plt.xticks(np.arange(1, lenth + 1, step=2), fontsize=20)
plt.yticks(fontsize=20)
plt.xlim(0, lenth + 1)
plt.xlabel('Epochs', fontsize=20)
plt.ylabel('Mean Square Error', fontsize=20)
plt.legend(fontsize=20)
plt.tight_layout()
path1 = 'F:/LSTM_Onsite_algorithm/Data_for_figures/Figure' + log_num
Path(path1).mkdir(parents=True, exist_ok=True)
plt.savefig(path1 + '/mse.svg')

# 保存训练过程的损失历史
np.savetxt(path1 + '/loss.txt', results.history['loss'], fmt='%.5f')
np.savetxt(path1 + '/val_loss.txt', results.history['val_loss'], fmt='%.5f')

# 写入日志文件
paras = ['CAV', 'CAVr', 'Tpd', 'Rhyp', 'Pa', 'Pv', 'Pd', 'Ia', 'Vs30']
dic = dict(Model=model_name, logs=log_num, Time_step=str(TIME_STEP), Dropout='False', Kernel_initializer='False',
           Units=str(units), Loss=loss, Optimizer=opt, Batchsize=str(batch_size), Epochs=str(epoch), Parameters=paras,
           sample_weight='False', buffer_size='False', Features='base e log', Label='base e log', Activation='None')
with open(path1 + '/' + model_name + '.txt', 'w') as f:
    # f.write(json.dumps(dic, indent=4, separators=(',', ':')))
    for key in dic.keys():
        if isinstance(dic[key], str):
            f.write(key.rjust(20, ' ') + '\t\t' + dic[key] + '\n')
        elif isinstance(dic[key], list):
            f.write(key.rjust(20, ' ') + '\t\t' + str(dic[key]) + '\n')
print('--------------------------------\nThe training process is complete\n--------------------------------')

# 读取测试集和震例的数据
X_test = np.loadtxt(path_data + test_x)
X_example = np.loadtxt(path_example + example_x)
X_test = np.reshape(X_test, [np.size(X_test, 0), TIME_STEP, FEATURES])
X_example = np.reshape(X_example, [np.size(X_example, 0), TIME_STEP, FEATURES])

# 检验模型在测试集上的性能
Pre_train = model.predict(X_train, verbose=1)
Pre_valid = model.predict(X_valid, verbose=1)
Pre_test = model.predict(X_test, verbose=1)
Pre_example = model.predict(X_example, verbose=1)

# 保存预测结果
np.savetxt(path_data + 'trainy_' + save_name + '.dat', Pre_train, fmt='%.6f')
np.savetxt(path_data + 'validy_' + save_name + '.dat', Pre_valid, fmt='%.6f')
np.savetxt(path_data + 'testy_' + save_name + '.dat', Pre_test, fmt='%.6f')
np.savetxt(path_example + 'casey_' + save_name + '.dat', Pre_example, fmt='%.6f')
