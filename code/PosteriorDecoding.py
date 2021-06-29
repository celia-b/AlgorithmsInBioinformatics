import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)

import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

# Sequence encoding
def encode(sequence, symbols):
    enc = [0] * len(sequence)
    
    for i in range(len(sequence)):
        enc[i] = symbols.find(sequence[i])
    
    return(enc)

# Load HMM (output from BaumWelch.py)
states = 2
symbols = "123456"
nsymbols = len(symbols)

initial_prob = [1.0/states, 1.0/states]
transition_matrix = np.asarray([0.887, 0.113, 0.045, 0.955]).reshape(2,2)
fair_prob = [0.168, 0.165, 0.155, 0.17, 0.172, 0.17] 
loaded_prob = [0.084, 0.091, 0.106, 0.085, 0.098, 0.536]  
emission_probs = [fair_prob, loaded_prob]

# Load sequences to decode
data_dir = "C:/Users/celib/Desktop/DENMARK/DTU/2-SEMESTER/ALGORITHMS_FOR_BIOINFORMATICS/data/Baum-Welch/"

sequence_file = data_dir + "casino.testseq"
states_file = data_dir + "casino.teststates"

with open(sequence_file) as handle:
    input_sequences = handle.read().split()

with open(states_file) as handle:
    input_states = handle.read().split()
    
input_encode = [encode(sequence, symbols) for sequence in input_sequences]


## Forward algorithm for HMM decoding
def initialize_forward(input_encode, states, initial_prob, emission_probs):
    
    alpha = np.zeros(shape=(states, len(input_encode), len(input_encode[0])))
        
    for i in range(0, states):                   # for each state
        
        for s in range(0, len(input_encode)):    # for each sequence
            
            alpha[i][s][0] = initial_prob[i]*emission_probs[i][input_encode[s][0]]
        
    return alpha

# Alpha is the table with the probabilities of observing an element x+1 in a state l considering all the possible pathways before.
alpha = initialize_forward(input_encode, states, initial_prob, emission_probs)

for s in range(0, len(input_encode)):            # for each sequence
    
    for i in range(1, len(input_encode[0])):     # steps in the chain
    
        for j in range(0, states):               # current state

            _sum = 0
        
            for k in range(0, states):           # previous state
            
                _sum += alpha[k][s][i-1] * transition_matrix[k][j]           
         
            # store prob
            alpha[j][s][i] = emission_probs[j][input_encode[s][i]] * _sum


## Backward algorithm for HMM decoding
def initialize_backward(input_encode, states):
    
    #beta = np.zeros(shape=(states, len(input_encode), dtype=float))
    beta = np.zeros(shape=(states, len(input_encode), len(input_encode[0])))
        
    for i in range(0, states):                   # for each state
        
        for s in range(0, len(input_encode)):    # for each sequence
  
            beta[i][s][-1] = 1
        
    return beta

beta = initialize_backward(input_encode, states)

# K = current state (j), L = next state (k), i = step in the chain (i)
for s in range(0, len(input_encode)):            # for each sequence
    
    for i in range(len(input_encode[0])-2, -1, -1): # steps in the chain
    
        for j in range(0, states):               # current state

            _sum = 0

            for k in range(0, states):           # next state
            
                _sum += emission_probs[k][input_encode[s][i+1]] * beta[k][s][i+1] * transition_matrix[j][k]
                
        
            # store prob
            beta[j][s][i] = _sum


## Posterior decoding
# posterior = f * b / p_x

posterior = np.zeros(shape=(len(input_encode),len(input_encode[0])), dtype=float)
p_state = 0
p_x = 0
p_x_vec = []

for s in range(0, len(input_encode)):            # for each sequence
    
    for j in range(0, states):                   # for each state
        p_x += alpha[j][s][-1]
        
    p_x_vec.append(p_x)                          # calculate the sequence prob for each of the sequences
    p_x = 0
   

# Select sequence of interest and get posterior probabilities
seq_select = 10                                   
posterior_plot = []

for s in range(0, len(input_encode)):            # for each sequence
    
    for i in range(0, len(input_encode[0])):     # steps in the chain
        posterior[s][i] = alpha[p_state][s][i]*beta[p_state][s][i]/p_x_vec[s] # p = (f_i * b_i)/p_x

        #print ("Posterior", i, s, input_sequences[s][i], input_encode[s][i], np.log(alpha[p_state, s, i]), np.log(beta[p_state, s, i]), posterior[s][i])
        
        if s == seq_select:                               # select sequence of interest
            posterior_plot.append(posterior[s][i])
  

# Plot 
posterior_plot = np.array(posterior_plot)
sequence_list = list(input_sequences[seq_select])
state = np.array([int(x) for x in list(input_states[seq_select])])

state0 = posterior_plot[np.where(state==0)]
state1 = posterior_plot[np.where(state==1)]

viterbi = ['1111000000000000000000000000000000000000']
viterbi1 = [int(x) for x in list(viterbi[0])]
for i in range(len(viterbi1)):
    if viterbi1[i] == 1:
        viterbi1[i] = 0
    else:
        viterbi1[i] = 0.1

plt.bar(np.where(state==0)[0], state0)
plt.bar(np.where(state==1)[0], state1)
plt.bar(range(len(viterbi1)), viterbi1, alpha=0.5, width=1, bottom=0.9, linewidth=0)
plt.axhline(y=0.5, color='black', linestyle='dashed')
plt.xticks(range(len(posterior_plot)), labels=sequence_list)
params = {'mathtext.default': 'regular' }          
plt.rcParams.update(params)
plt.ylabel('$P(S_i = fair | X)$')
plt.show()