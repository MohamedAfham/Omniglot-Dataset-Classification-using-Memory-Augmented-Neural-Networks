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
K = 2 #num_samples_per_class
B = 100
total = N*K*B
manncell = MANNCell(rnn_size_list = 128, memory_size = 20,memory_vector_dim = 4,head_num =2)
state = manncell.zero_state(total,tf.float32)
read_vector_list = state['read_vector_list']

data_generator = DataGenerator(num_classes = N,num_samples_per_class=K)
train_data = data_generator.sample_batch('train',B)
val_data = data_generator.sample_batch('val',25)
inp = Input(shape=(784,))

x = train_data[0]
x_arr = [np.array(i) for i in x]
x_tf = tf.constant(np.array(x_arr))

y_true = tf.constant(np.array([np.array(i) for i in train_data[1]]))

y_pred,next_state = manncell(inp, state)
state = next_state
y_pred = tf.slice(y_pred,[0,0],[total,N])
adam = tf.train.AdamOptimizer(learning_rate = 1.5)

with tf.Session() as sess:  
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())
    model = Model(inputs=inp,outputs=y_pred)
    model.compile(optimizer = adam,loss = 'categorical_crossentropy')
        