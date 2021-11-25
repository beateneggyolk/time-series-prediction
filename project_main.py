%matplotlib inline
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd
import os

from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Input, Dense, GRU, Embedding
from tensorflow.python.keras.optimizers import RMSprop
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau

tf.__version__
tf.keras.__version__
pd.__version__

import weather

weather.maybe_download_and_extract()
cities = weather.cities
cities 

df = weather.load_resampled_data()
df.head()

df['Esbjerg']['Pressure'].plot()
df['Roskilde']['Pressure'].plot()
df.values.shape

df.drop(('Esbjerg', 'Pressure'), axis=1, inplace=True)
df.drop(('Roskilde', 'Pressure'), axis=1, inplace=True)
df.values.shape
df.head(1)
df['Odense']['Temp']['2006-05':'2006-07'].plot()
df['Aarhus']['Temp']['2006-05':'2006-07'].plot()
df['Roskilde']['Temp']['2006-05':'2006-07'].plot()
df['Various', 'Day'] = df.index.dayofyear
df['Various', 'Hour'] = df.index.hour

target_city = 'Odense'
target_names = ['Temp', 'WindSpeed', 'Pressure']
shift_days = 1
shift_steps = shift_days * 24 .

df_targets = df[target_city][target_names].shift(-shift_steps)
df[target_city][target_names].head(shift_steps + 5)
df_targets.head(5)
df_targets.tail()

x_data = df.values[0:-shift_steps]
print(type(x_data))
print("Shape:", x_data.shape)

y_data = df_targets.values[:-shift_steps]
print(type(y_data))
print("Shape:", y_data.shape)

num_data = len(x_data)
num_data
train_split = 0.9
num_train = int(train_split * num_data)
num_train
num_test = num_data - num_train
num_test

x_train = x_data[0:num_train]
x_test = x_data[num_train:]
len(x_train) + len(x_test)

y_train = y_data[0:num_train]
y_test = y_data[num_train:]
len(y_train) + len(y_test)

num_x_signals = x_data.shape[1]
num_x_signals

num_y_signals = y_data.shape[1]
num_y_signals

print("Min:", np.min(x_train))
print("Max:", np.max(x_train))

x_scaler = MinMaxScaler()
x_train_scaled = x_scaler.fit_transform(x_train)

print("Min:", np.min(x_train_scaled))
print("Max:", np.max(x_train_scaled))

x_test_scaled = x_scaler.transform(x_test)

y_scaler = MinMaxScaler()
y_train_scaled = y_scaler.fit_transform(y_train)
y_test_scaled = y_scaler.transform(y_test)

print(x_train_scaled.shape)
print(y_train_scaled.shape)

def batch_generator(batch_size, sequence_length):
    
while True:
        x_shape = (batch_size, sequence_length, num_x_signals)
        x_batch = np.zeros(shape=x_shape, dtype=np.float16)
        y_shape = (batch_size, sequence_length, num_y_signals)
        y_batch = np.zeros(shape=y_shape, dtype=np.float16)

         for i in range(batch_size):
            idx = np.random.randint(num_train - sequence_length)
            x_batch[i] = x_train_scaled[idx:idx+sequence_length]
            y_batch[i] = y_train_scaled[idx:idx+sequence_length]
        yield (x_batch, y_batch)

sequence_length = 24 * 7 * 8
sequence_length
generator = batch_generator(batch_size=batch_size,
                            sequence_length=sequence_length)
x_batch, y_batch = next(generator)
print(x_batch.shape)

print(y_batch.shape)

seq = x_batch[batch, :, signal]
plt.plot(seq)

seq = y_batch[batch, :, signal]
plt.plot(seq)

validation_data = (np.expand_dims(x_test_scaled, axis=0),
                   np.expand_dims(y_test_scaled, axis=0))
model = Sequential()
model.add(GRU(units=512,
              return_sequences=True,
              input_shape=(None, num_x_signals,)))
model.add(Dense(num_y_signals, activation='sigmoid'))
if False:
    from tensorflow.python.keras.initializers import RandomUniform
    init = RandomUniform(minval=-0.05, maxval=0.05)
    model.add(Dense(num_y_signals,
                    activation='linear',
                    kernel_initializer=init))
warmup_steps = 50
def loss_mse_warmup(y_true, y_pred):

    y_true_slice = y_true[:, warmup_steps:, :]
    y_pred_slice = y_pred[:, warmup_steps:, :]
    loss = tf.losses.mean_squared_error(labels=y_true_slice,
                                        predictions=y_pred_slice)
    loss_mean = tf.reduce_mean(loss)
    return loss_mean
optimizer = RMSprop(lr=1e-3)

model.compile(loss=loss_mse_warmup, optimizer=optimizer)
model.summary()

path_checkpoint = '23_checkpoint.keras'
callback_checkpoint = ModelCheckpoint(filepath=path_checkpoint,
                                      monitor='val_loss',
                                      verbose=1,
                                      save_weights_only=True,
                                      save_best_only=True)
callback_early_stopping = EarlyStopping(monitor='val_loss',
                                        patience=5, verbose=1)

callback_tensorboard = TensorBoard(log_dir='./23_logs/',
                                   histogram_freq=0,
                                   write_graph=False)

callback_reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                       factor=0.1,
                                       min_lr=1e-4,
                                       patience=0,
                                       verbose=1)

callbacks = [callback_early_stopping,
             callback_checkpoint,
             callback_tensorboard,
             callback_reduce_lr]

%%time
model.fit_generator(generator=generator,
                    epochs=20,
                    steps_per_epoch=100,
                    validation_data=validation_data,
                    callbacks=callbacks)

try:
    model.load_weights(path_checkpoint)
except Exception as error:
    print("Error trying to load checkpoint.")
    print(error)

result = model.evaluate(x=np.expand_dims(x_test_scaled, axis=0),
                        y=np.expand_dims(y_test_scaled, axis=0))

print("loss (test-set):", result)

if False:
    for res, metric in zip(result, model.metrics_names):
        print("{0}: {1:.3e}".format(metric, res))

def plot_comparison(start_idx, length=100, train=True):
    if train:
        x = x_train_scaled
        y_true = y_train
    else:
        x = x_test_scaled
        y_true = y_test
    end_idx = start_idx + length
    
    x = x[start_idx:end_idx]
    y_true = y_true[start_idx:end_idx]
    x = np.expand_dims(x, axis=0)
    y_pred = model.predict(x)
    
    y_pred_rescaled = y_scaler.inverse_transform(y_pred[0])
    
    for signal in range(len(target_names)):
        signal_pred = y_pred_rescaled[:, signal]
        signal_true = y_true[:, signal]
        plt.figure(figsize=(15,5))
        plt.plot(signal_true, label='true')
        plt.plot(signal_pred, label='pred')
        p = plt.axvspan(0, warmup_steps, facecolor='black', alpha=0.15)
        plt.ylabel(target_names[signal])
        plt.legend()
        plt.show()
plot_comparison(start_idx=100000, length=1000, train=True)
plot_comparison(start_idx=200000, length=1000, train=True)

df['Odense']['Temp'][200000:200000+1000].plot()
df_org = weather.load_original_data()
df_org.xs('Odense')['Temp']['2002-12-23':'2003-02-04'].plot()

plot_comparison(start_idx=200, length=1000, train=False)
