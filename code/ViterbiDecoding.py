import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)

import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

## ------------------------ ##
## This program uses the Viterbi algorithm to predict the most
## probable underlying set of states for a given sequence of dice rolls
## given the HMM that generated them.
## ------------------------- ##

# Parse from stdin
parser = ArgumentParser(description="Viterbi decoding")
parser.add_argument("-f",action="store", type=str, dest="sequence_file", help="File containing sequences")
parser.add_argument("-f",action="store", type=str, dest="states_file", help="File containing underlying states")

# Parameter data structure initialization
def initialize(encode_sequence, states, initial_prob, transition_matrix, emission_probs):
    
    delta = np.zeros(shape=(states, len(encode_sequence), len(encode_sequence[0])))
    
    arrows = np.ndarray(shape=(states, len(encode_sequence), len(encode_sequence[0])), dtype=object)
    
    # initial conditions
    for i in range(0, states): # for each state
        for s in range(0, len(encode_sequence)): # for each sequence
  
            delta[i][s][0] = initial_prob[i] + emission_probs[i][encode_sequence[s][0]] # Remember we work in log space 

            arrows[i][s][0] = 0
    
    return delta, arrows

# Encode dice rolls
def encode(sequence, symbols):
    
    enc = [0] * len(sequence)
    
    for i in range(len(sequence)):
        enc[i] = symbols.find(sequence[i])
    
    return(enc)


# Load HMM (output from BaumWelch.py)
states = 2
symbols = "123456"
nsymbols = len(symbols)

initial_prob = np.log10([1.0/states, 1.0/states])
transition_matrix = np.log10(np.asarray([0.911, 0.089, 0.053, 0.947]).reshape(2,2))
fair_prob = np.log10([0.182, 0.169, 0.166, 0.165, 0.17, 0.147]) 
loaded_prob = np.log10([0.082, 0.111, 0.091, 0.104, 0.107, 0.505])  
emission_probs = [fair_prob, loaded_prob]


# Load sequences to decode
sequence_file = parser.parse_args().sequence_file
data_dir = "C:/Users/celib/Desktop/DENMARK/DTU/2-SEMESTER/ALGORITHMS_FOR_BIOINFORMATICS/data/"

sequence_file = data_dir + "Baum-Welch/" + sequence_file
states_file = data_dir + "Baum-Welch/" + states_file

with open(sequence_file) as handle:
    input_sequences = handle.read().split()

with open(states_file) as handle:
    input_states = handle.read().split()
    
input_encode = [encode(sequence, symbols) for sequence in input_sequences]


## Markov chain simulation
# Now we initialize the system, where the probability of starting at state1 = probability of starting at state2
# Delta is the table with the Viterbi acumulated log(P)s.
# The arrows are for backtracking
delta, arrows = initialize(input_encode, states, initial_prob, transition_matrix, emission_probs)

for s in range(0, len(input_sequences)): # For each sequence
    
    for i in range(1, len(input_sequences[0])): # Steps in the chain

        for j in range(0, states): # Current state

            # We fill each cell in delta with the Viterbi formula:
            # log(Pl(i+1)) = log(pl(i+1)) + max(log(Pk(i)) + log(akl)) 
            max_arrow_prob = -np.inf # A very low negative number
            max_arrow_prob_state = -1

            for k in range(0, states): # Previous state

                # arrow_prob is the probability of ending in the state j from the state k
                # Max score in previous state + transition from previous to current state
                arrow_prob = delta[k][s][i-1] + transition_matrix[k][j] # There are 2 possible arrow_probs for each current state

                if arrow_prob > max_arrow_prob: 
                    max_arrow_prob = arrow_prob
                    max_arrow_prob_state = k

            # store prob:
            # emission probability of current step + max(score in previous step + transition probability from previous tu current step)
            delta[j][s][i] = emission_probs[j][input_encode[s][i]] + max_arrow_prob

            # store arrow
            arrows[j][s][i] = max_arrow_prob_state


## Backtracking
paths = []

for s in range(len(input_sequences)): # For each sequence
    path = []
    
    # We backtrack from the last position with the state that has the highest probability
    max_state = np.argmax(delta[:, s, -1])
    max_value = delta[max_state, s, -1]
    
    print("log(Max_path):\t", max_value)
    print("Sequence:\t", input_sequences[s])
    
    old_state = max_state
    path.append(str(max_state))
    
    # Go from the last to the first step of the chain
    for i in range(len(input_encode[s])-2, -1, -1):

        current_state = arrows[old_state][s][i+1]

        path.append(str(current_state))

        old_state = current_state 
    
    print("Predicted path:\t", "".join(reversed(path)))
    print("True path:\t", input_states[s])
    
    paths.append("".join(reversed(path)))


## Get errors and plot as histogram
errors = []
index = 0
for s in range(len(input_states)):
    match = 0
    true_path = list(input_states[s])
    predicted_path = list(paths[s])
    for i in range(len(true_path)):
        if true_path[i] == predicted_path[i]:
            match += 1
    error = 1 - match/len(true_path)
    #print("Sequence", index, ":\t", input_sequences[s])
    #print("Predicted path:\t", paths[s])
    #print("True path:\t", input_states[s])
    #print("Error:\t", "{:.1f}".format(error*100), "%")
    errors.append(error)
    index += 1


num_bins = 10
n, bins, patches = plt.hist(errors, num_bins, facecolor='red', alpha=0.5)
plt.xlabel("Error")
plt.ylabel("Number of sequences (%)")
plt.show()