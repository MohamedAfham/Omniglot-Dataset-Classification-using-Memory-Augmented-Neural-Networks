import numpy as np 
import tensorflow.compat.v1 as tf 
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Dense, Input,LSTM,Reshape,Conv2D,Flatten
from load_data import DataGenerator
import matplotlib.pyplot as plt

N = 10
K = 2
B = 15
data_generator = DataGenerator(num_classes=N,num_samples_per_class=K)

#model = Sequential()
inp = Input(shape=(K*N,784 + N))
y_pred = LSTM(units=128,activation='tanh',return_sequences=True)(inp)
y_pred = Dense(10,activation='softmax')(y_pred)
y_pred = y_pred[:,-1,:N]

adam = tf.keras.optimizers.Adam(learning_rate = 0.001)
model = Model(inputs=inp,outputs=y_pred)
model.compile(optimizer = adam,loss = 'categorical_crossentropy',metrics = ['accuracy'])

history = model.fit_generator(data_generator.sample_batch('train',B),epochs=100,steps_per_epoch=50,validation_data = data_generator.sample_batch('val',B),validation_steps = B)
print(model.evaluate_generator(data_generator.sample_batch('test',B),steps = 50))
fig ,axs = plt.subplots(nrows = 2,ncols=1,figsize=(6,4))
axs[0].plot(history.history['accuracy'])
axs[0].plot(history.history['val_accuracy'])
axs[0].set_title('Model accuracy')
axs[0].set_ylabel('Accuracy')
axs[0].set_xlabel('Epoch')
axs[0].legend(['Train', 'Val'], loc='upper left')

axs[1].plot(history.history['loss'])
axs[1].plot(history.history['val_loss'])
axs[1].set_title('Model loss')
axs[1].set_ylabel('Loss')
axs[1].set_xlabel('Epoch')
axs[1].legend(['Train', 'Val'], loc='upper left')
fig.savefig("Accuracy Loss Plot LSTM.jpg")
plt.show()