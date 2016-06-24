

"""LSTM for time series classification
Made: 30 march 2016

This model takes in time series and class labels.
The LSTM models the time series. A fully-connected layer
generates an output to be classified with Softmax
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.python.framework import ops
from tensorflow.python.ops import clip_ops
import freeze_graph
import os

def sample_batch(X_train,y_train,batch_size,num_steps):
    """ Function to sample a batch for training"""
    N,data_len = X_train.shape
    ran  = np.random.choice(N-batch_size,1)
    ind_N = np.arange(ran,ran+batch_size)
    ind_start = 0 # ysuzuki added 2016/06/03
    #form batch
    X_batch = X_train[ind_N,ind_start:ind_start+num_steps]
    y_batch = y_train[ind_N]
    return X_batch,y_batch

def sample_abn_batch(X_train,y_train,batch_size,num_steps):
    """ Function to sample a batch for training"""
    N,data_len = X_train.shape
    ran  = np.random.choice(N-num_steps*batch_size,1)
    ind_N = np.arange(ran,ran+num_steps*batch_size,num_steps)
    #print(ind_N)
    ind_start = 0 # ysuzuki added 2016/06/03
    #form batch
    X_batch = X_train[ind_N,ind_start:ind_start+num_steps]
    #print(X_batch)
    y_batch = y_train[ind_N]
    return X_batch,y_batch
    
def sample_batch_test(X_train,y_train,batch_size,num_steps):
    """ Function to sample a batch for training"""
    N,data_len = X_train.shape
    #ran  = np.random.choice(N-batch_size,1)
    ran  = 0;
    ind_N = np.arange(ran,ran+batch_size)
    ind_start = 0 # ysuzuki added 2016/06/03
    #form batch
    X_batch = X_train[ind_N,ind_start:ind_start+num_steps]
    y_batch = y_train[ind_N]
    return X_batch,y_batch

def check_test(X_test,y_test,batch_size,num_steps):
    """ Function to check the test_accuracy on the entire test set
    This is a workaround. I haven't figured out yet how to make the graph
    general for multiple batch sizes."""
    N = X_test.shape[0]
    num_batch = np.floor(N/batch_size)
    test_acc = np.zeros(num_batch)
    
    for i in range(int(num_batch)):
      
      X_batch, y_batch = sample_batch(X_test,y_test,batch_size,num_steps)
      test_acc[i] = session.run(accuracy,feed_dict = {input_data: X_batch, targets: y_batch, initial_state:state,keep_prob:1})
    return np.mean(test_acc)

"""Hyperparamaters"""
init_scale = 0.08           #Initial scale for the states
max_grad_norm = 25          #Clipping of the gradient before update
num_layers = 2              #Number of stacked LSTM layers
num_steps = 16              #Number of steps to backprop over at every batch
hidden_size = 13            #Number of entries of the cell state of the LSTM
max_iterations = 100         #Maximum iterations to train
batch_size = 15             #batch size
outputclass = 2             #2016/06/01 ysuzuki added
unit = 10                    #2016/06/01 ysuzuki added
ntest = 100                 #num of calculate cost and accuracy

#for saving graph
checkpoint_prefix = os.path.join("models/", "saved_checkpoint")
checkpoint_state_name = "checkpoint_state"
input_graph_name = "input_graph.pb"
output_graph_name = "output_graph.pb"


"""Place holders"""
input_data = tf.placeholder(tf.float32, [None, num_steps], name = 'input_data')
targets = tf.placeholder(tf.int64, [None], name='Targets')
#Used later on for drop_out. At testtime, we pass 1.0
keep_prob = tf.placeholder("float", name = 'Drop_out_keep_prob') 

initializer = tf.random_uniform_initializer(-init_scale,init_scale)
with tf.variable_scope("model", initializer=initializer):
  """Define the basis LSTM"""
  with tf.name_scope("LSTM_setup") as scope:
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size)
    if (keep_prob is not None): # add ysuzuki 2016/06/01
        #if keep_prob < 1: 
        lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=keep_prob)
    cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * num_layers) #up indent 2016/06/07 ysuzuki
    initial_state = cell.zero_state(batch_size, tf.float32)   #Initialize the zero_state. Note that it has to be run in session-time
    #We have only one input dimension, but we generalize our code for future expansion
    inputs = tf.expand_dims(input_data, 2)

  #Define the recurrent nature of the LSTM
  #Re-use variables only after first time-step
  with tf.name_scope("LSTM") as scope:
    outputs = []
    state = initial_state
    with tf.variable_scope("LSTM_state"):
     for time_step in range(num_steps):
     #for time_step in range(batch_size):
       if time_step > 0: tf.get_variable_scope().reuse_variables()
       #print(type(inputs))
       #print ( inputs[:, :, :]).shape
       #s = raw_input("stop")
       (cell_output, state) = cell(inputs[:, time_step, :], state)
       outputs.append(cell_output)       #Now cell_output is size [batch_size x hidden_size]

#Generate a classification from the last cell_output
#Note, this is where timeseries classification differs from sequence to sequence
#modelling. We only output to Softmax at last time step
with tf.name_scope("Softmax") as scope:
  with tf.variable_scope("Softmax_params"): 
    softmax_w = tf.get_variable("softmax_w", [hidden_size, outputclass]) # suzuki 2016/06/01                 
    softmax_b = tf.get_variable("softmax_b", [outputclass]) # suzuki 2016/06/01                              

  logits = tf.matmul(cell_output, softmax_w) + softmax_b
  #Use sparse Softmax because we have mutually exclusive classes    
  loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits,targets,name = 'Sparse_softmax')
  cost = tf.reduce_sum(loss,name="costvalue") / batch_size
  #Pass on a summary to Tensorboard
  cost_summ = tf.scalar_summary('Cost',cost)
  # Calculate the accuracy
  correct_prediction = tf.equal(tf.argmax(logits,1), targets)
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"), name= 'accu')
  accuracy_summary = tf.scalar_summary("accuracy", accuracy)

"""Optimizer"""
with tf.name_scope("Optimizer") as scope:
  tvars = tf.trainable_variables()
  #We clip the gradients to prevent explosion
  grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),max_grad_norm)
  optimizer = tf.train.AdamOptimizer(8e-3)
  gradients = zip(grads, tvars)
  train_op = optimizer.apply_gradients(gradients,name = 'train_op')
  # Add histograms for variables, gradients and gradient norms.
  # The for-loop loops over all entries of the gradient and plots
  # a histogram. We cut of
  for gradient, variable in gradients:
    if isinstance(gradient, ops.IndexedSlices):
      grad_values = gradient.values
    else:
      grad_values = gradient
    h1 = tf.histogram_summary(variable.name, variable)
    h2 = tf.histogram_summary(variable.name + "/gradients", grad_values)
    h3 = tf.histogram_summary(variable.name + "/gradient_norm", clip_ops.global_norm([grad_values]))
  
"""Load the data"""
dummy = False
if dummy:
  #data_train = np.loadtxt('UCR_TS_Archive_2015/Two_Patterns/Two_Patterns_TRAIN',delimiter=',')
  #data_test_val = np.loadtxt('UCR_TS_Archive_2015/Two_Patterns/Two_Patterns_TEST',delimiter=',')
  data_train = np.loadtxt('data/train_new',delimiter=',')
  data_test_val = np.loadtxt('data/test_new',delimiter=',')
else:
  data_train_norm = np.loadtxt('data/TRAIN_batch_norm1000',delimiter=',')
  data_train_abn = np.loadtxt('data/TRAIN_batch_abn_first_second',delimiter=',')
  data_test_val = np.loadtxt('data/TEST_batch2000',delimiter=',')
data_test,data_val = np.split(data_test_val,2)


X_train_norm = data_train_norm[:,1:]
X_train_abn = data_train_abn[:,1:]
X_val = data_val[:,1:]
X_test = data_test[:,1:]
N = X_train_norm.shape[0]
Ntest = X_test.shape[0]
# Targets have labels 1-indexed. We subtract one for 0-indexed
y_train_norm = data_train_norm[:,0]
y_train_abn = data_train_abn[:,0]
y_val = data_val[:,0]
y_test = data_test[:,0]


#Final code for the TensorBoard
merged = tf.merge_all_summaries()


# Collect the costs in a numpy fashion
epochs = np.floor(batch_size*max_iterations / N)
print('Train with approximately %d epochs' %(epochs))
perf_collect = np.zeros((3,int(np.floor(max_iterations /unit))))
perf_collect_local = np.zeros((3,int(np.floor(unit))))



"""Session time"""
with tf.Session() as session:

  writer = tf.train.SummaryWriter("/home/suzuki/LSTM_tsc-master/log", session.graph_def)
  
  tf.initialize_variables(tf.all_variables(), name='init_all_vars_op2').run()
  #tf.initialize_all_variables().run()

  step = 0
  step_local = 0

  for i in range(max_iterations):
    saver = tf.train.Saver()

    # Calculate some sizes
    N_norm= X_train_norm.shape[0]
    
    #Sample batch for training
    X_batch, y_batch = sample_batch(X_train_norm,y_train_norm,batch_size,num_steps)
    state = initial_state.eval()  #Fire up the LSTM
    
    #Next line does the actual training
    session.run(train_op,feed_dict = {input_data: X_batch,targets: y_batch,initial_state: state,keep_prob:0.5})
    
    X_batch, y_batch = sample_abn_batch(X_train_abn,y_train_abn,batch_size,num_steps)
    state = initial_state.eval()  #Fire up the LSTM
    #print(X_batch, y_batch)
    #Next line does the actual training
    session.run(train_op,feed_dict = {input_data: X_batch,targets: y_batch,initial_state: state,keep_prob:0.5})

    #print("iteration #",i+1,"/",max_iterations)
    if i==0:
        # Uset this line to check before-and-after test accuracy
        acc_test_before = check_test(X_test,y_test,batch_size,num_steps)
    if i%unit == 0:
        #Evaluate training performance
        cost_train_out_sum = 0.0
        for j in range(ntest):
            X_batch, y_batch = sample_batch(X_train_norm,y_train_norm,batch_size,num_steps)                           
            cost_out = session.run(cost,feed_dict = {input_data: X_batch, targets: y_batch, initial_state:state,keep_prob:1})
            cost_train_out_sum += cost_out
        perf_collect[0,step] = cost_train_out_sum/float(ntest)
            #print('At %d out of %d train cost is %.3f' %(i,max_iterations,cost_out)) #Uncomment line to follow train cost

        #Evaluate validation performance
        cost_out_sum = 0.0
        acc_val_sum = 0.0
        for j in range(ntest):
            #X_batch, y_batch = sample_batch(X_val,y_val,batch_size,num_steps)
            X_batch, y_batch = sample_batch_test(X_val,y_val,batch_size,num_steps)
            #print(X_batch)
            result = session.run([cost,merged,accuracy,loss,softmax_w],feed_dict = {input_data: X_batch, targets: y_batch, initial_state:state,keep_prob:1})
            cost_out_sum += result[0]
            acc_val_sum += result[2]
            lost_tmp = result[3]
            w_tmp = result[4]

        perf_collect[1,step] = cost_out_sum/float(ntest)
        perf_collect[2,step] = acc_val_sum/float(ntest)

        step +=1
        acc_test = check_test(X_test,y_test,batch_size,num_steps)

        print('At %d out of %d val cost is %.3f and val acc is %.3f' %(i+1,max_iterations,cost_out_sum/float(ntest),acc_val_sum/float(ntest)))
        
        #summary_str += result[1]
        #Write information to TensorBoard
        #writer.add_summary(summary_str, i)
        #writer.flush()

    if i == (max_iterations - unit):
        saver.save(session, checkpoint_prefix, global_step=0, latest_filename=checkpoint_state_name)
        tf.train.write_graph(session.graph.as_graph_def(), "models/", input_graph_name)    
        tf.train.write_graph(session.graph.as_graph_def(), "models/", "test2.pb", as_text=True)    
        print("i = ",i)
        print("w = ",result[4])
        print("loss = ",result[3])
        print("cost = ",result[0])
        print("accuracy = ",result[2])


  input_graph_path = os.path.join("models/", input_graph_name)
  input_saver_def_path = ""
  input_binary = False
  input_checkpoint_path = os.path.join("models/", 'saved_checkpoint') + "-0"

  # Note that we this normally should be only "output_node"!!!
  output_node_names = "Softmax/Sparse_softmax,Softmax/costvalue,Softmax/accu,Softmax_params/softmax_w"
  #output_node_names = "Softmax/Sparse_softmax/Sparse_softmax"
  restore_op_name = "save/restore_all"
  filename_tensor_name = "save/Const:0"
  output_graph_path = os.path.join("models/", output_graph_name)
  clear_devices = False

  freeze_graph.freeze_graph(input_graph_path, input_saver_def_path,input_binary, input_checkpoint_path,output_node_names, restore_op_name,filename_tensor_name, output_graph_path,clear_devices, "")


  tf.train.write_graph(session.graph.as_graph_def(), "models/", "test.pb", as_text=True)
    
"""Additional plots"""
print('The accuracy on the test data is %.3f, before training was %.3f' %(acc_test,acc_test_before))

xnum = np.arange(0,max_iterations,unit)
plt.plot(xnum,perf_collect[0],"-o",label='Train error')
plt.plot(xnum,perf_collect[1],"-o",label = 'Valid')
plt.plot(xnum,perf_collect[2],"-o",label = 'Valid accuracy')
plt.axis([0, max_iterations, 0, 1.3*(np.max(perf_collect))])
plt.xlabel("iteration", fontsize=20)
plt.ylabel("Error / Accuracy ", fontsize=20)
plt.legend()
plt.show()




