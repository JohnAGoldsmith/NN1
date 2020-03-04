import numpy as np
import math 

def sigma(x):
    return 1.0/(1+ math.exp(-1*x))
def dsigma(x):
     return x * (1-x)
 

input_size = 1000
output_size = input_size
hidden_layer_size = 25
learning_rate = 0.1

np.set_printoptions(precision=3)

# ======================
# initialize
# ======================
M1 = np.random.rand(input_size *  hidden_layer_size).reshape(input_size, hidden_layer_size)
M2 = np.random.rand(hidden_layer_size * output_size).reshape(hidden_layer_size, output_size)
Input = np.zeros(input_size)
Input[5] = 1.0


hidden_layer_input = np.zeros(hidden_layer_size)    # Hlayer is activation of hidden layer
hidden_layer_activation = np.zeros(hidden_layer_size)    # Hlayer is activation of hidden layer
hidden_layer_activation[0] = 1.0  # the bias term
hidden_layer_error = np.zeros(hidden_layer_size)

output_layer_input = np.zeros(output_size)
output_layer_activation = np.zeros(output_size)

error = np.zeros(output_size)
truth = np.zeros(output_size)
# ======================
# compute hidden layer
# ======================

hidden_layer_input = np.matmul(M1.transpose(), Input)
for row in range(hidden_layer_size):
    hidden_layer_activation[row] = sigma(hidden_layer_input[row])
    
# ======================
# compute output layer
# ======================


output_layer_input = np.matmul(M2.transpose(), hidden_layer_activation)
for row in range(output_size):
    output_layer_activation[row] = sigma(output_layer_input[row])
  
# ======================
# compute error layer
# ======================
error = output_layer_activation - truth
 
for hidden in range(hidden_layer_size):
    for out in range(output_size):
        delta = hidden_layer_activation[hidden] * error[out] * dsigma(output_layer_activation[out])   
        M2[hidden,out] -= delta * learning_rate
        #print hidden, out, delta
        hidden_layer_error += delta

for in_unit in range(input_size):
    for out_unit in range(hidden_layer_size):
        delta = Input[in_unit] *  hidden_layer_error[out_unit] * dsigma(hidden_layer_activation[out_unit])
        M1[in_unit,out_unit] -= delta * learning_rate
        #print in_unit, out_unit, delta


