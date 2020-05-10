import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Dense, Input,LSTM,Reshape
from tensorflow.keras.optimizers import Adam
from UpdateState import state_update
import numpy as np
import matplotlib.pyplot as plt
from load_data import DataGenerator
from MANN import MANNCell

N = 10 #num_samples
K = 2 #num_samples_per_class
B = 15
total = N*K*B
manncell = MANNCell(rnn_size = 128, memory_size = 20,memory_vector_dim = 4,head_num = 2,samples_per_batch=N*K)
state = manncell.zero_state(B,tf.float32)

data_generator = DataGenerator(num_classes = N,num_samples_per_class=K)

inp = Input(shape=(K*N,784 + N))

controller_output,next_state = manncell(inp, state)

with tf.Session() as sess:  
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())
    y_pred_memory = tf.concat([controller_output] + next_state['read_vector_list'], axis=2)
    state = next_state
    y_pred_memory = Dense(N,activation='softmax')(y_pred_memory)
    y_pred = y_pred_memory[:,-1,:N]
    adam = tf.keras.optimizers.Adam(learning_rate = 0.001)


    model = Model(inputs=inp,outputs=y_pred)
    model.compile(optimizer = adam,loss = 'categorical_crossentropy',metrics = ['accuracy'])
    history = model.fit_generator(data_generator.sample_batch('train',B),epochs=150,steps_per_epoch=50,validation_data = data_generator.sample_batch('val',B),validation_steps = B)
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
    fig.savefig("Accuracy Loss Plot MANN.jpg")
    plt.show()