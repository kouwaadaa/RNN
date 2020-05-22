import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import csv

class EarlyStopping():
    def __init__(self,patience=0, verbose=0):
        self._step=0
        self._loss=float('inf')
        self.patience= patience
        self.verbose = verbose
    
    def validate(self,loss):
        if self._loss <loss:
            self._step += 1
            if self._step >self.patience:
                if self.verbose:
                    print('early stopping')
                return True
        else:
            self._step = 0
            self._loss = loss
        return False
    

#def model
def inference(x, n_batch, maxlen=None, n_hidden=None, n_out=None):
    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.01)
        return tf.Variable(initial)

    def bias_variable(shape):
        initial = tf.zeros(shape, dtype=tf.float32)
        return tf.Variable(initial)

    cell = tf.nn.rnn_cell.BasicRNNCell(n_hidden)
    initial_state = cell.zero_state(n_batch, tf.float32)

    state = initial_state
    outputs=[]
    with tf.variable_scope('RNN'):
        for t in range(maxlen):
            if t > 0:
                tf.get_variable_scope().reuse_variables()
            (cell_output, state)=cell(x[:,t,:],state)
            outputs.append(cell_output)

    output = outputs[-1]

    V = weight_variable([n_hidden, n_out])
    c = bias_variable([n_out])
    y = tf.matmul(output, V) + c

    return y

#def error
def loss(y, t):
    mse = tf.reduce_mean(tf.square(y-t))
    return mse

#def algo
def training(loss):
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.999)

    train_step = optimizer.minimize(loss)
    return train_step


if __name__ == '__main__':
    #data preparation
    xlabel = []
    data_a = []
    target = []
    data = []
    
    def get_data(filename):
        with open(filename, 'r') as f:
            csvFileReader = csv.reader(f,delimiter=",")
            next(csvFileReader)
            for row in csvFileReader:
                xlabel.append(row[0])
                data_a.append(float(row[1]))
        return 

            
    get_data('data2018.csv')
    length_of_sequences = len(data_a)
    maxlen = 24
    
    for i in range(0, length_of_sequences - maxlen): #length_of_sequences - maxlen + 1
        data.append(data_a[i:i+maxlen])
        target.append(data_a[i+maxlen])
    
    X = np.array(data).reshape(len(data), maxlen, 1)
    Y = np.array(target).reshape(len(data), 1)
    
    N_train = int(len(data)*0.9)
    N_validation = len(data) - N_train
    
    #X_train,X_validation,Y_train,Y_validation = train_test_split(X, Y, test_size=N_validation)
    
    #N_train = int(len(data) * 0.9)
    X_train = X[:N_train]
    X_validation = X[N_train:]
    Y_train = Y[:N_train]
    Y_validation = Y[N_train:]

    
    #model Configuration
    n_in = len(X[0][0]) #1 
    n_hidden = 20
    n_out = len(Y[0]) #1 
    
    x = tf.placeholder(tf.float32, shape=[None, maxlen, n_in])
    t = tf.placeholder(tf.float32, shape=[None, n_out])
    n_batch = tf.placeholder(tf.int32,shape=[])
    
    y = inference(x, n_batch, maxlen=maxlen, n_hidden=n_hidden, n_out=n_out)
    loss = loss(y, t)
    train_step=training(loss)
    
    #model learning
    epochs = 500
    batch_size = 10
    
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    
    n_batches = N_train // batch_size
    
    history = {
            'val_loss': [],
            'val_acc': []
            }
    early_stopping = EarlyStopping(patience=10,verbose=1)
    
    for epoch in range(epochs):
        X_,Y_  = shuffle(X_train,Y_train)
        
        for i in range(n_batches):
            start = i * batch_size
            end = start + batch_size
            
            sess.run(train_step, feed_dict={
                    x: X_[start:end],
                    t: Y_[start:end],
                    n_batch: batch_size
            })
    
        val_loss = loss.eval(session=sess, feed_dict={
                x: X_validation,
                t: Y_validation,
                n_batch: N_validation
        })
        
        history['val_loss'].append(val_loss)
        print('epoch:',epoch,'validation loss:', val_loss)
        
        if early_stopping.validate(val_loss):
            break
       
    #graf
    truncate = maxlen
    Z = X[:1] #first data
    
    original = np.array([data_a[i] for i in range(maxlen)])
    predicted = [None for i in range(maxlen)]
    
    for i in range(length_of_sequences - maxlen + 1): #length_of_sequences - maxlen + 1
        z_ = Z[-1:] #last data
        y_ = y.eval(session=sess, feed_dict={
                x: Z[-1:],
                n_batch:1
                })
    
        sequence_ = np.concatenate(
                (z_.reshape(maxlen, n_in)[1:],y_),axis=0).reshape(1, maxlen, n_in)
        Z = np.append(Z, sequence_, axis=0)
        predicted.append(y_.reshape(-1))
    predicted = np.array(predicted)
    
    plt.rc('font', family='serif')
    plt.figure()
    #plt.ylim([-5,35])
    plt.plot(data_a,linestyle='dotted',color='#aaaaaa')
    plt.plot(original, linestyle='dashed',color='black')
    plt.plot(predicted, color='black')
    plt.show()
    
    
