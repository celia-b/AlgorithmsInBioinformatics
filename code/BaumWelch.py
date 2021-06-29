import numpy as np
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(palette='pastel')

## ------------------------ ##
## This program uses the Baum-Welch algorithm to optimize the parameters
## of the HMM that describes the given data.

## The data file, seed, number of iterations and other parameters can be specified in the commandline.
## ------------------------- ##

# Encode a series of dice rolls
def encode(sequence, symbols):
    
    enc = [0] * len(sequence)
    
    for i in range(len(sequence)):
        enc[i] = symbols.find(sequence[i])
    
    return(enc)

# Forward algorithm for HMM decoding
def forward(input_encode, states, transition_matrix, emission_probs, initial_prob):
    
    alpha = np.zeros(shape=(states, len(input_encode)))
    scale = np.zeros(shape=(len(input_encode)))
    
    # First column 
    alpha[:,0] = initial_prob * emission_probs[:, input_encode[0]]
    # Rescale 
    scale[0] = np.sum(alpha[:, 0])
    alpha[:, 0] = alpha[:, 0] / scale[0]
    
    # Other columns
    for i in range(1, len(input_encode)): # Positions in the sequence
        for j in range(0, states): # Current state
            
                alpha[j, i] = emission_probs[j][input_encode[i]] * transition_matrix[:,j] @ alpha[:,i-1] # Sum over previous states
            
        # Rescale    
        scale[i] = np.sum(alpha[:, i]) 
        alpha[:, i] = alpha[:, i] / scale[i]

    return alpha, scale

# Backward algorithm for HMM decoding
def backward(input_encode, states, transition_matrix, emission_probs):
    
    beta = np.zeros(shape=(states, len(input_encode)))
    scale = np.zeros((len(input_encode)))
    
    # Last column
    beta[:, -1] = 1
    # Rescale
    scale[-1] = np.sum(beta[:, -1])
    beta[:, -1] = beta[:, -1]/scale[-1]

    for i in range(len(input_encode)-2, -1, -1): # Positions in the sequence
    
        for j in range(0, states): # Current state
            # store prob
            beta[j, i] = emission_probs[:, input_encode[i+1]] * beta[:, i + 1] @ transition_matrix[j, :] # Sum over next states
            
        # Rescale beta
        scale[i] = np.sum(beta[:, i])
        beta[:, i] = beta[:, i]/scale[i]
    
    return beta, scale

# Baum-Welch algorithm for unsupervised HMM parameter optimization
def baum_welch(input_encode, states, transition_matrix, emission_matrix, alpha, beta):
    
    # Calculate epsilon, gamma and sequence probabilities
    epsilon = np.zeros((states, states, len(input_encode[:, 0]) - 1, len(input_encode[0, :])))
    
    for r in range(len(input_encode[0, :])):            # Loop through sequences
        
        for i in range(len(input_encode[:, 0]) - 1):    # Loop through positions in sequence
            for j in range(states):                     # Next state
                for k in range(states):                 # Current state
                    
                    # Calculating sequence probabilities with scaled alpha/beta 
                    # At position i of sequence r, we can build the sequence going (i -> i+1)
                    # from 0 to 0 + from 0 to 1 + from 1 to 0 + from 1 to 1 
                    denominator = 0
                    # Loop through states for denominator
                    for q in range(states):             # Current state
                        for p in range(states):         # Next state
                            denominator += alpha[p, i, r] * transition_matrix[p, q] * emission_matrix[q, input_encode[i+ 1, r]] * beta[q, i + 1, r] # A bit different because of the rescaling (we can't sum the last column anymore)
                            
                    epsilon[j, k, i, r] = alpha[k, i, r] * transition_matrix[k, j] * emission_matrix[j, input_encode[i+ 1, r] ] * beta[j, i + 1, r]/ denominator
    gamma = np.sum(epsilon, axis = 0) # Sum over "next state" for each "current state"
    
    # Calculate new transition probabilities 
    # First sum is over positions in sequence (i)
    # Second sum is over sequences, which we do in numerator and denominator to get an average (r)
    transition_matrix = (np.sum(np.sum(epsilon, axis = 2), axis = 2) / np.sum(np.sum(gamma, axis = 1), axis = 1)).T
    
    # Last gamma cannot be calculated from summing epsilons
    denominator = 0
    for j in range(states):
        denominator += alpha[j, -1, :] * beta[j, -1, :] # Again weird bc of the rescaling
    last_gamma = np.reshape(alpha[:, -1, :] * beta[:, -1, :] / denominator, (states, 1, len(input_encode[0, :])))
    gamma = np.concatenate((gamma, last_gamma) , axis = 1)
    
    # Calculate new emission probabilities
    emission_matrix = np.zeros((np.shape((emission_matrix))))
    for i in range(np.shape((emission_matrix))[1]):             # Loop through emission probabilities
        nominator = 0
        for r in range(len(input_encode[0, :])):                # Loop through sequences
            nominator += np.sum(gamma[:, np.array(list(input_sequences[r])) == str(i+1), r ], axis = 1)
            
        emission_matrix[:, i] = nominator / np.sum(np.sum(gamma, axis = 1), axis = 1)
        
    return transition_matrix, emission_matrix


## Main code
# Parse from stdin
parser = ArgumentParser(description="Baum-Welch HMM")
parser.add_argument("-f",action="store", type=str, dest="sequence_file", help="File containing sequences", )
parser.add_argument("-st",action="store", type=int, dest="states", default=2, help="Number of states (default = 2)", )
parser.add_argument("-se",action="store", type=int, dest="seed", help="Seed for reproducibility", )
parser.add_argument("-i",action="store", type=int, dest="n_iter", default = 1000,  help="Number of iterations (default 1000)", )
parser.add_argument("-e", action="store", type=str, dest="symbols", default= "123456", help="Symbols used in sequence as str (default = '123456')")

# Define variables
data_dir = "C:/Users/celib/Desktop/DENMARK/DTU/2-SEMESTER/ALGORITHMS_FOR_BIOINFORMATICS/data/"
sequence_file = parser.parse_args().sequence_file
states = parser.parse_args().states
n_iter = parser.parse_args().n_iter
seed = parser.parse_args().seed
symbols = parser.parse_args().symbols

# Define initial probablities
np.random.seed(seed)

initial_prob = [1/states] * states
transition_matrix = np.random.rand(states,states)
fair_prob = np.random.rand(len(symbols))
loaded_prob = np.random.rand(len(symbols))

# Normalize so rows sum to 1
initial_prob = initial_prob / np.sum(initial_prob)
transition_matrix = transition_matrix / np.sum(transition_matrix, axis = 1)[:, np.newaxis]
fair_prob = fair_prob/np.sum(fair_prob)
loaded_prob = loaded_prob/np.sum(loaded_prob)

emission_matrix = np.array([fair_prob, loaded_prob])

# Collect structures for plotting probabilities
training_prob = []

with open(data_dir + sequence_file) as handle:
    input_sequences = handle.read().split()


# Only works with sequences of equal length, but could be updated to work with variable length 
for _ in range(n_iter):
    # Initialize structures
    alpha = np.zeros((states, len(input_sequences[0]), len(input_sequences))) # extra dimension for each sequence (as many alphas and betas as sequences)
    beta = np.zeros(shape=(states, len(input_sequences[0]), len(input_sequences)))
    input_encode = np.zeros((len(input_sequences[0]), len(input_sequences)), dtype=int)
    scale_alpha = np.zeros((len(input_sequences[0]), len(input_sequences)))
    scale_beta = np.zeros((len(input_sequences[0]), len(input_sequences)))
    
    for i in range(len(input_sequences)):
        input_encode[:, i] = encode(input_sequences[i], symbols)
        # Calculate alpha and beta by forward and backwards
        alpha[:, :, i], scale_alpha[:, i] = forward(input_encode[:, i], states, transition_matrix, emission_matrix, initial_prob)
        beta[:, :, i], scale_beta[:, i] = backward(input_encode[:, i], states, transition_matrix, emission_matrix)
    training_prob.append(np.mean(np.sum(np.log(scale_alpha), axis = 0))) # The probability now changes because of the rescaling
    # Update transition and emission probabilities using baum welch
    transition_matrix, emission_matrix = baum_welch(input_encode, states, transition_matrix, emission_matrix, alpha, beta)


## Plot of training probabilities
plt.plot(range(n_iter), training_prob)
plt.xlabel("N iterations")
plt.ylabel("log(P)")
plt.show()


## Optimized parameters
print('Transition Matrix:')
transition_matrix = np.around(transition_matrix, 3)
for i in range(transition_matrix.shape[0]):
    print(*transition_matrix[i,:], sep='\t',)
    
print('Emission_matrix:')
emission_matrix = np.around(emission_matrix, 3)
for i in range(emission_matrix.shape[0]):
    print(*emission_matrix[i,:], sep='\t')