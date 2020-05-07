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
tot = N*K*B
manncell = MANNCell(rnn_size_list = 128, memory_size = 20,memory_vector_dim = 4,head_num =2)
state = manncell.zero_state(tot,tf.float32)
read_vector_list = state['read_vector_list']

data_generator = DataGenerator(num_classes = N,num_samples_per_class=K)
train_data = data_generator.sample_batch('train',B)
val_data = data_generator.sample_batch('val',25)

x = train_data[0]
x_arr = [np.array(i) for i in x]
x_tf = tf.constant(np.array(x_arr))

y_true = tf.constant(np.array([np.array(i) for i in train_data[1]]))
for epoch in range(10):
    NTM_output,state = manncell(x_tf, state)
    y_pred = Reshape((1,136))(NTM_output)
    y_pred = LSTM(64,return_sequences=False)(y_pred)
    y_pred = Dense(5,activation=tf.keras.layers.Softmax())(y_pred)
    optimizer = tf.train.AdamOptimizer(learning_rate = 1.5)
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    updated = optimizer.minimize(loss(y_true,y_pred))
    with tf.Session() as sess:  
        sess.run(tf.local_variables_initializer())
        sess.run(tf.global_variables_initializer())
        ls = loss(y_true,y_pred).eval()
        
        pred = y_pred.eval()
        true = y_true.eval()
        true_label = [np.where(i==1)[0].tolist()[0] for i in true]
        pred_label = [np.where(i==np.max(i))[0].tolist()[0] for i in pred]
        bool_list = []
        for i in range(len(true_label)):
            if pred_label[i]==true_label[i]:
                bool_list.append(0)
            else:
                bool_list.append(1)
        count_1 = bool_list.count(1)
        accuracy = (tot-count_1)/tot
        print ("Epoch:",epoch+1,"------- Loss:",ls,"---------------------------- Accuracy:",accuracy) 