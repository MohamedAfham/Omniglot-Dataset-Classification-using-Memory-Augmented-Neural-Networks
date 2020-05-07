import numpy as np 
import tensorflow.compat.v1 as tf 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input,LSTM,Reshape

class MANNCell():
    def __init__(self, rnn_size_list, memory_size, memory_vector_dim, head_num, gamma=0.95,
                 reuse=True):

        #initialize all the variables
        self.rnn_size_list = rnn_size_list
        self.memory_size = memory_size
        self.memory_vector_dim = memory_vector_dim
        self.head_num = head_num                                   
        self.reuse = reuse
        self.step = 0
        self.gamma = gamma
      
        
        #initialize controller as the basic rnn cell\
        network = Sequential()
        network.add(LSTM(units=128,activation='tanh',return_sequences=True))
        network.add(LSTM(units=64,activation='tanh',return_sequences=True))
        network.add(Dense(5,activation='softmax'))

        self.controller = network

    def __call__(self, x, prev_state):
        prev_read_vector_list = prev_state['read_vector_list']   
        controller_input = tf.concat([x] + prev_read_vector_list, axis=-1)


        #next we pass the controller, which is the RNN cell, the controller_input and prev_controller_state
        controller_input = tf.expand_dims(controller_input,axis=1)
        controller_output = self.controller(controller_input)
        controller_output = tf.squeeze(controller_output)
            
        num_parameters_per_head = self.memory_vector_dim + 1
        total_parameter_num = num_parameters_per_head * self.head_num
        
        #Initiliaze weight matrix and bias and compute the parameters
        weights = tf.Variable(tf.random_normal([controller_output.get_shape()[1], total_parameter_num], stddev=0.35))
        biases = tf.Variable(tf.zeros([total_parameter_num]))
        parameters = tf.nn.xw_plus_b(controller_output, weights, biases)
        head_parameter_list = tf.split(parameters, self.head_num, axis=1)
        
        
        #previous read weight vector
        prev_w_r_list = prev_state['w_r_list']   
        
        #previous memory
        prev_M = prev_state['M']
        
        #previous usage weight vector
        prev_w_u = prev_state['w_u']
        
        #previous index and least used weight vector
        prev_indices, prev_w_lu = self.least_used(prev_w_u)
        
        #read weight vector
        w_r_list = []
        
        #write weight vector
        w_w_list = []
        
        #key vector
        k_list = []
    
        #now, we will initialize some of the important parameters that we use for addressing.     
        for i, head_parameter in enumerate(head_parameter_list):
            with tf.variable_scope('addressing_head_%d' % i):
                
                #key vector
                k = tf.tanh(head_parameter[:, 0:self.memory_vector_dim], name='k')
                
                #sig_alpha
                sig_alpha = tf.sigmoid(head_parameter[:, -1:], name='sig_alpha')
                
                #read weights
                w_r = self.read_head_addressing(k, prev_M)
                
                #write weights
                w_w = self.write_head_addressing(sig_alpha, prev_w_r_list[i], prev_w_lu)
           
            w_r_list.append(w_r)
            w_w_list.append(w_w)
            k_list.append(k)
            

        #usage weight vector 
        w_u = self.gamma * prev_w_u + tf.add_n(w_r_list) + tf.add_n(w_w_list)   

        #update the memory
        M_ = prev_M * tf.expand_dims(1. - tf.one_hot(prev_indices[:, -1], self.memory_size), dim=2)
        
        
        #write operation
        M = M_
        with tf.variable_scope('writing'):
            for i in range(self.head_num):
                
                w = tf.expand_dims(w_w_list[i], axis=2)
                k = tf.expand_dims(k_list[i], axis=1)
                M = M + tf.matmul(w, k)

        #read opearion
        read_vector_list = []
        with tf.variable_scope('reading'):
            for i in range(self.head_num):
                read_vector = tf.reduce_sum(tf.expand_dims(w_r_list[i], dim=2) * M, axis=1)
                read_vector_list.append(read_vector)       

        
        #controller output
        NTM_output = tf.concat([controller_output] + read_vector_list, axis=1)

        state = {
           # 'controller_state': controller_state,
            'read_vector_list': read_vector_list,
            'w_r_list': w_r_list,
            'w_w_list': w_w_list,
            'w_u': w_u,
            'M': M,
        }

        self.step += 1
        return NTM_output, state

    #weight vector for read operation   
    def read_head_addressing(self, k, prev_M):
        
        "content based cosine similarity"
        
        k = tf.expand_dims(k, axis=2)
        inner_product = tf.matmul(prev_M, k)
        k_norm = tf.sqrt(tf.reduce_sum(tf.square(k), axis=1, keep_dims=True))
        M_norm = tf.sqrt(tf.reduce_sum(tf.square(prev_M), axis=2, keep_dims=True))
        norm_product = M_norm * k_norm
        K = tf.squeeze(inner_product / (norm_product + 1e-8))                  
        K_exp = tf.exp(K)
        w = K_exp / tf.reduce_sum(K_exp, axis=1, keep_dims=True)               
        
        return w
    
    #weight vector for write operation
    def write_head_addressing(self,sig_alpha, prev_w_r_list, prev_w_lu):
        prev_w_r = prev_w_r_list[-1]
        return sig_alpha * prev_w_r + (1. - sig_alpha) * prev_w_lu     
    
    #least used weight vector
    def least_used(self,w_u):
        _, indices = tf.nn.top_k(w_u, k=self.memory_size)
        w_lu = tf.reduce_sum(tf.one_hot(indices[:, -self.head_num:], depth=self.memory_size), axis=1)
        return indices, w_lu

    
    #next we define the function called zero state for initializing all the states - 
    #controller state, read vector, weights and memory
    def zero_state(self,batch_size,dtype):
        one_hot_weight_vector = np.zeros([batch_size, self.memory_size])
        one_hot_weight_vector[..., 0] = 1
        one_hot_weight_vector = tf.constant(one_hot_weight_vector, dtype=tf.float32)
        with tf.variable_scope('init', reuse=self.reuse):
            state = {
                'read_vector_list': [tf.zeros([batch_size, self.memory_vector_dim])
                                     for _ in range(self.head_num)],
                'w_r_list': [one_hot_weight_vector for _ in range(self.head_num)],
                'w_u': one_hot_weight_vector,
                'M': tf.constant(np.ones([batch_size, self.memory_size, self.memory_vector_dim]) * 1e-6, dtype=tf.float32)
            }
            return state