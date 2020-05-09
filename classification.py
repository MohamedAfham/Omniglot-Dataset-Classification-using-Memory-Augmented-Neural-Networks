import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Dense, Input,LSTM,Reshape
from tensorflow.keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt
from load_data import DataGenerator
from MANN import MANNCell

N = 5 #num_samples
K = 1 #num_samples_per_class
B = 15
total = N*K*B
manncell = MANNCell(rnn_size = 128, memory_size = 20,memory_vector_dim = 4,head_num =2)
state = manncell.zero_state(total,tf.float32)
read_vector_list = state['read_vector_list']

data_generator = DataGenerator(num_classes = N,num_samples_per_class=K)

inp = Input(shape=(784 + N,))

y_pred,next_state = manncell(inp, state)
state = next_state
y_pred = tf.slice(tf.squeeze(y_pred),[0,0],[total,N])
adam = tf.train.AdamOptimizer(learning_rate = 0.001)

with tf.Session() as sess:  
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())
    model = Model(inputs=inp,outputs=y_pred)
    model.compile(optimizer = adam,loss = 'categorical_crossentropy',metrics = ['accuracy'])
    history = model.fit_generator(data_generator.sample_batch('train',B),epochs=50,steps_per_epoch=50,validation_data = data_generator.sample_batch('val',B),validation_steps = B)
    fig ,ax = plt.subplots(figsize=(8,4))
    ax.plot(history.history['accuracy'])
    ax.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    fig.savefig("Accuracy Plot.jpg")
    plt.show()
    model.save('MANNModel')