import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Dense, Input,LSTM,Reshape
from tensorflow.keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt
from load_data import DataGenerator
from MANN import MANNCell

N = 6 #num_samples
K = 1 #num_samples_per_class
B = 15
total = N*K*B
manncell = MANNCell(rnn_size_list = [128,64], memory_size = 20,memory_vector_dim = 4,head_num =2,samples_per_batch=N*K)
state = manncell.zero_state(B,tf.float32)
read_vector_list = state['read_vector_list']

data_generator = DataGenerator(num_classes = N,num_samples_per_class=K)

inp = Input(shape=(K*N,784 + N))

y_pred,next_state = manncell(inp, state)
state = next_state
y_pred = y_pred[:,-1,:N]
adam = tf.keras.optimizers.Adam(learning_rate = 0.0002)

with tf.Session() as sess:  
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())
    model = Model(inputs=inp,outputs=y_pred)
    model.compile(optimizer = adam,loss = 'categorical_crossentropy',metrics = ['accuracy'])
    history = model.fit_generator(data_generator.sample_batch('train',B),epochs=100,steps_per_epoch=50,validation_data = data_generator.sample_batch('val',B),validation_steps = B)
    print(model.evaluate_generator(data_generator.sample_batch('test',B),steps = 50))