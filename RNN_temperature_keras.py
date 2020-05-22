import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers.recurrent import SimpleRNN
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
import csv

np.random.seed(0)

def generate_temperatures(filename):
    xlabel = []
    temperatures = []
    with open(filename, 'r') as f:
            csvFileReader = csv.reader(f,delimiter=",")
            next(csvFileReader)
            for row in csvFileReader:
                xlabel.append(row[0])
                temperatures.append(float(row[1]))
    temperatures = np.array(temperatures)
    temperatures = np.reshape(temperatures, (temperatures.size))
    return temperatures


def generate_data(temperatures, length_per_unit, dimension):
    sequences = []
    target = []
    for i in range(0, temperatures.size - length_per_unit):
        sequences.append(temperatures[i:i + length_per_unit])
        target.append(temperatures[i + length_per_unit])

    X = np.array(sequences).reshape(len(sequences), length_per_unit, dimension)
    Y = np.array(target).reshape(len(sequences), dimension)

    N_train = int(len(sequences) * 0.9)
    X_train = X[:N_train]
    X_validation = X[N_train:]
    Y_train = Y[:N_train]
    Y_validation = Y[N_train:]

    return (X_train, X_validation, Y_train, Y_validation)


def build_model(input_shape, hidden_layer_count):
    model = Sequential()
    model.add(SimpleRNN(hidden_layer_count, input_shape=input_shape))
    model.add(Dense(input_shape[1]))
    model.add(Activation('linear'))
    model.compile(loss='mse', optimizer=Adam())
    return model


# 一つの時系列データの長さ
LENGTH_PER_UNIT = 24
# 一次元データを扱う
DIMENSION = 1
# 年別月平均気温の生成
temperatures = generate_temperatures('data2018.csv')
# トレーニング、バリデーション用データの生成
X_train, X_validation, Y_train, Y_validation = generate_data(temperatures, LENGTH_PER_UNIT, DIMENSION)

# SimpleRNN隠れ層の数
HIDDEN_LAYER_COUNT = 25
# 入力の形状
input_shape=(LENGTH_PER_UNIT, DIMENSION)
# モデルの生成
model = build_model(input_shape, HIDDEN_LAYER_COUNT)

# モデルのトレーニング
epochs = 500
batch_size = 10
early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1)
model.fit(X_train, Y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(X_validation, Y_validation),
          callbacks=[early_stopping])

# 予測を行う
part_of_sequence = np.array([temperatures[i] for i in range(LENGTH_PER_UNIT)])
predicted = [None for i in range(LENGTH_PER_UNIT)]
Z = X_train[:1, :, :]
for i in range(temperatures.size - LENGTH_PER_UNIT + 1):
    y_ = model.predict(Z)
    # 予測結果を入力として利用するため、第0項を削除し予測結果をひっつける
    Z = np.concatenate(
            (Z.reshape(LENGTH_PER_UNIT, DIMENSION)[1:], y_),
            axis=0).reshape(1, LENGTH_PER_UNIT, DIMENSION)
    predicted.append(y_.reshape(-1))
predicted = np.array(predicted)

# 予測結果の描画
plt.rc('font', family='serif')
plt.figure()
plt.ylim([0.0, 35.0])
plt.plot(temperatures, linestyle='dotted', color='#aaaaaa')
plt.plot(part_of_sequence, linestyle='dashed', color='black')
plt.plot(predicted, color='black')
plt.show()